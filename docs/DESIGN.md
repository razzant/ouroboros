# Ouroboros Design System

Normative authority for the **visual and interaction semantics** of the Ouroboros
UI: what a size means, what a colour claims, what a chip is allowed to say.

Authority split:

- **This file** decides semantics — type scale, hierarchy, foreground and state
  colour, status conventions, card/row anatomy, density.
- **`docs/DEVELOPMENT.md` → `## Design System`** decides the engineering rules
  that keep those semantics intact — where values may live, which component is
  the SSOT, what counts as review debt, how a visual change is verified.

Shared values and control recipes live in `web/ui.css`, loaded before page
styles by both the SPA and the served onboarding document. Page styles own
composition, not another copy of the shared palette. This file names roles;
it does not copy an inventory.

The shell offers **Settings → Appearance → Light / Dark / System**. New clients
start on System; explicit Light or Dark remains pinned. The shared semantic
palettes in `web/ui.css` preserve geometry and status meanings. Light uses white
reading surfaces, dark text and an opaque header, with decorative matrix hidden.

Appearance belongs to a browser profile or desktop client, not an account or
server setting. Existing saved Light/Dark choices keep their meaning. Storage
failures are visible; clearing site data returns the choice to System. Switching
repaints mounted charts and diagrams without rebuilding views or losing drafts.
Independent iframe interiors remain author-owned, not automatically recoloured.
Desktop persistence requires a launcher built with persistent WebView storage;
server restart alone cannot verify survival across full quit/relaunch. Mechanism
and deployment limits: ARCHITECTURE §3 “Navigation and shared UI contracts”.

---

## 1. Type scale

Four sizes. There is no fifth.

| Token | Value | Role |
| --- | --- | --- |
| `--type-meta` | 12px | Labels, notes, chips, timing/quota lines, captions |
| `--type-body` | 14px | Default reading text, values, controls, row titles |
| `--type-section` | 16px | Section and card titles |
| `--type-page` | 24px | Page / wizard step title, and display text (a login code) |

Line heights: `--line-meta` (1.35) for short meta lines, `--line-body` (1.5) for
prose, `--line-title` (1.3) for headings.

Rules:

- A value with no exact token **rounds to the nearest token**. It never mints a
  fifth size. 13px becomes `--type-body`, 15px becomes `--type-section`, 11px
  becomes `--type-meta`.
- **No new raw `10px` / `11px` text on a migrated surface.** Below 12px the
  dark theme forces a choice between illegible and glaring, and the glaring
  option is what produced the owner's "too much small high-contrast white text"
  report. `tests/test_web_typography_static.py` enforces this.
- Control chrome keeps its own dimension tokens (`--button-font-size`,
  `--pill-font-size`). They are control geometry, not the reading scale; do not
  replace them with type tokens or vice versa.

## 2. Hierarchy rule

**In any row, card or field, exactly one thing is primary.** Everything else
steps down. Concretely:

- **Label** → `--type-meta` in `--text-meta`, sentence case.
- **Value** (the thing the owner came to read or change) → `--type-body` in
  `--text-primary`.
- **Meta** (last run, quota, effort, timing, provenance) → `--type-meta` in
  `--text-meta`.
- **Section title** → `--type-section` semibold; **subsection heading** →
  `--type-body` semibold. A bare `<h3>`/`<h4>`/`<strong>` that inherits the
  browser default is a defect: it lands at bold 16px full white and ties with,
  or beats, the content it introduces.

**The 12px UPPERCASE label pattern is retired** on migrated surfaces. All-caps
at a small size costs legibility, widens every label, and when a panel repeats
it dozens of times the labels collectively out-shout the values they describe.
Authored label strings should read as sentence case; CSS must not manufacture
caps with `text-transform`.

## 3. Foreground and state colour

| Token | Meaning |
| --- | --- |
| `--text-primary` | The one thing this row/card is about; interactive control labels |
| `--text-meta` | Real secondary content: labels, notes, hints, meta lines |
| `--text-secondary` | A quieter step below meta, for a dense repeated field |
| `--text-disabled` | Genuinely inert or incidental content only |

`--text-muted` is a legacy alias of `--text-disabled`; new work names
`--text-meta` or `--text-disabled` so the intent is readable in the diff.

`--text-secondary` is a real fourth step, not an alias: it sits between meta
and disabled and is written at ~50 call sites. Reach for it only when
`--text-meta` is genuinely too loud — a value repeated on every row of a dense
list, a chip's supporting word — and `--text-disabled` would be unreadable.
The criterion is the same one that governs the whole table: **if the owner has
to read it to act, it is `--text-meta` or brighter.** A load-bearing caveat, a
one-off note, a hint that explains a control: those stay at meta. A tone or an
unclassifiable ink is what produced the original "too much small
high-contrast white text" report from the opposite direction, so a rule that
declares a size and no colour is still the worst of the options.

Two failure modes this table exists to prevent, both observed in this codebase:

- **Secondary content parked at the disabled foreground.** A load-bearing
  caveat rendered at `--text-disabled` reads as greyed-out chrome and gets
  skipped. If the owner is meant to read it, it is `--text-meta`.
- **Secondary content with no foreground at all**, inheriting `--text-primary`.
  This is the loudest of all failures because it is invisible in the CSS — the
  rule simply declares a size and says nothing about colour.

### Brand accent

The brand red is **one value**, `--accent`. Every appearance of it is either
that token, the named roles built on it (`--accent-light` for text on dark,
`--accent-dim` for a fill, `--focus-accent-border` / `--focus-accent-ring` for
focus), or a rung of the accent alpha ladder (`--accent-04` … `--accent-65`).
A new alpha is a rung added to the ladder, never an `rgba()` literal in a rule:
the ladder is what makes "make the accent calmer" a one-line change instead of
a grep.

The app and first-run wizard consume the same palette source. A shared accent
change updates its named roles and alpha ladder there; neither page shadows
those roles with its own values. `tests/test_web_typography_static.py` checks
the actual stylesheet links, nonempty shared roles and per-document variable
resolution. The SM1 browser oracle also checks both rendered documents after
restart, so a missing link or a page-local override cannot pass as consistency.
A wizard-only layout token is fine; a second value for a shared role is not.

### Focus

Keyboard focus has **one** appearance:

```css
outline: 2px solid var(--focus-accent-border);
outline-offset: 2px;
```

`outline-offset: -2px` is the only sanctioned variant, for a control that sits
flush inside a strip that clips an outer ring (sidebar rows, header buttons).
Nothing else: not a blue ring, not a green one, not a `box-shadow` standing in
for an outline, and not a colour picked to match the control it is on. A focus
ring is the reader's cursor; if it changes colour per surface, it stops
reading as one thing.

**Hover paint is not a focus ring.** A rule written as
`.x:hover, .x:focus-visible { background: … }` gives a keyboard user exactly
what a mouse user gets by accident and nothing that says "you are here". Where
a control wants both, the shared paint stays in the hybrid rule and the ring
goes in a `:focus-visible`-only rule of its own.

Text fields are the exception, and they keep their own established idiom:
`border-color: var(--focus-accent-border)` plus
`box-shadow: 0 0 0 3px var(--focus-accent-ring)`. A field already has a border
to recolour, so an outline outside it would be a second frame.

### Controls and editable choices

Text, number, password, select and multiline fields use the same `.ui-control`
family; `.ui-checkbox` keeps native checkbox behavior. `.ui-field` groups a
label, control and optional `.ui-field-help`. A placeholder is an example,
never the field's only name. Help and validation belong to that field without
changing the alignment of neighboring controls and their actions.

Short fixed choices keep native selects, including the platform's own popup.
A control never widens its column: a select shows its chosen label on one
line, clipped at its own edge, and the full label stays in the platform's
popup.
Model selection uses the shared editable chooser: suggestions assist typing
without becoming an allowlist. A saved unknown model remains editable; a
catalog refresh preserves the real input, selection and composition. Escape
or blur closes suggestions without assigning a value. Selected, hover, focus,
disabled and invalid states have different meanings and remain distinguishable.

Tabs expose one selected view and one keyboard entry point. Arrow keys and
Home/End move through available tabs; restoring a selected tab reveals it by
scrolling its strip, without moving the page or taking focus. Menus and
editable suggestion lists share viewport placement, not keyboard semantics:
a menu moves focus among actions, a chooser keeps it in the input. Dialog
focus stays in the modal context and returns on close when the caller remains
available. Popups sit outside decorative clipping and fit the usable viewport.

### `.muted`

`.muted` is a **colour-only utility**: `color: var(--text-meta)`, nothing else.
It must never set `font-size`. It is written at ~50 call sites that already
sized themselves, and a size here would silently resize all of them. A scoped
rule (`.some-context .muted`) still wins on specificity where a surface needs a
local variant.

### Dark-theme contrast

WCAG 4.5:1 is a **floor, not a target**. On near-black, pushing small text
toward pure white causes halation — the glyphs bloom, and because everything is
maximally bright, nothing is emphasised. The result is a screen that is
simultaneously harder to read and flatter in hierarchy.

**De-emphasise the secondary rather than amplifying the primary.** When
something needs to stand out, drop the ink around it, do not raise its own. All
Primary (15.9:1) and meta (9.2:1) clear the 4.5:1 floor against `--bg-primary`
with room to spare. `--text-disabled` is deliberately BELOW it (3.5:1) and is
therefore reserved for genuinely disabled or incidental content, which WCAG
exempts; it must never carry meaning a reader has to obtain.

### A selected state is not exempt from contrast

A **status hue** and a **status foreground** are different values, and the
selected state of a control must use the foreground. Selected Advisory in the
enforcement group read `--amber` (`#f59e0b`) over a 12% amber wash: ~2:1 on the
light surface, unreadable exactly when the owner had chosen it. The rule that
closes this: a selected control tints with the `--status-*-fg` /
`--status-*-bg` / `--status-*-border` triple, which is defined per theme, and
never with the raw hue token, which is not.

The same reasoning covers images. A colour baked into a `data:` URI cannot be
themed, because a custom property cannot be interpolated into the URI string —
which is why the select chevron was a pale `#e2e8f0` on white. **The whole
image is the token** (`--select-arrow`), overridden per theme, not the colour
inside it. `tests/test_appearance_static.py` holds both facts.

## 4. Status and chips

A status has **an explicit foreground/background pair**, never a foreground
derived from whatever generic opacity happens to sit on the element.

Status, owner action, and urgent notification are separate product concepts:

- **Status** states a fact about the affected object. It does not imply that the
  owner can or must act. Task status uses one factual word family: `Working`,
  `Done`, `Done with warnings`, `Failed`, `Cancelled`. The same five words are
  the host's durable label vocabulary for its own task rows in Main and the
  Project thread (`OUTCOME_PHASE_HEADLINE`), not only the browser's.
- **Owner action** exists only when the responsible domain exposes a current
  concrete continuation, such as Resume, Retry, Connect, Repair, Grant access,
  or Restart. The action is a real adjacent control; severity alone never
  manufactures one.
- **Urgent notification** is a rare, time-sensitive interruption. It uses the
  product's explicit incident/notification seam, not a red status or a failed
  task as a proxy.

A task-bound `Reviews` history row may be the only retained fact for its owner.
That row keeps a neutral owner anchor visible, but hides task status and typing
until a real task status or activity arrives; review presence alone never means
`Working`, `Done`, or owner attention.

A host fact about a task is a row of that task's card, never a standalone
bubble beside it. A reviewer panel that settles after its task already ended
adds one System row naming the verdict and which revision it covered; that row
lands inside the finished card (its Reviews group carries the note, the timeline
keeps the row) without changing the card's chip, title or meta, and a standalone
row appears only when the task has no card record in the page. The untyped
terminal host notice and the origin-addressed routing notices stay ordinary rows by design. Local diagnostic failures remain inspectable
in details and Logs, but do not relabel the whole still-working task. A failed child keeps a compact factual
`Failed` marker inside its parent while the root continues under its own
authoritative status. Internal reason codes belong in details and diagnostics,
not compact headlines. Where a card does show a cause, it says it in the owner's
words while the record keeps the machine code; a cause with no sentence yet stays
raw rather than borrowing a wrong one. The routing receipt under an owner
message is such a surface: a refused addressing act carries the host-composed
`cause` sentence (`project_dialogue.routing_refusal_cause` — one host table for
the receipt line, the System row and the picker toast), a landed act carries
none, and an unknown reason stays raw. A terminal whose preserved output was
never reviewed shows that output labelled rather than hidden: a short labelled
excerpt beside the pointer to the full copy, so a `Failed` card over applied work
is never a bare headline and never names preserved bytes without a way to reach
them. Where a stop receipt already carries the same text
in the very chat the card is written to, the card keeps the label and the
pointer alone; a card in another chat keeps the excerpt.

| Role | Foreground | Background | Border |
| --- | --- | --- | --- |
| Success / connected | `--status-ok-fg` | `--status-ok-bg` | `--status-ok-border` |
| Warning / degraded | `--status-warn-fg` | `--status-warn-bg` | `--status-warn-border` |
| Error / failed | `--status-error-fg` | `--status-error-bg` | `--status-error-border` |
| Neutral / classification | `--status-neutral-fg` | `--status-neutral-bg` | `--status-neutral-border` |

- **Status renders as dot + text.** The dot carries the state at a glance, so
  the sentence does not have to shout it in saturated colour and can sit at
  ordinary reading contrast.
- **Neutral is a real state**, not an absence of one. A classification chip
  (which agent, which family) is neutral: it is a tag, not an alarm.
  A tone value the code actually emits (`muted`) must have a rule; falling
  through to a default is how chips end up white.
- Chips are `--type-meta`, not smaller, and are not uppercased.
- `--green` / `--amber` / `--red` are the saturated hues, and they are for
  things that are not text: dots, switch tracks, progress. The `--status-*-fg`
  tints are for text on near-black; do not swap them. (There was also a
  `--tone-ok` / `--tone-warn` / `--tone-danger` alias family, plus
  `--accent-task` / `--accent-system` / `--accent-user` / `--accent-project`
  and `--ui-tone-*`. They were named here and referenced by nothing at all, so
  every surface kept inventing its own literal instead. They are gone; the
  vocabulary above is the whole vocabulary.)

Current completion and independent criticism are separate facts. An informed Advisory
author finish may complete the current subject while its original review remains
FAIL or DEGRADED; the old critic alone must not paint that completion Failed.
Independent execution, artifact, verification or publication failures still apply.
Blocking corrections saved without fresh approval and an explicit unfinished stop
remain unaccepted; show the retained work and reason through the existing five-word
status family and details, without inventing reviewer PASS or a new status badge.

### The tone primitive

Two shapes carry a tone, and they are not interchangeable:

- **A status sentence** — `.ui-status[data-tone]`. Foreground only, rendered as
  dot + text. Filling it would turn every inline status into a badge and make
  "connected" the loudest thing on the panel.
- **A status chip** — `.ui-chip[data-tone]`. The full triple, because the chip
  *is* the status and has nothing else to carry it.

Surfaces whose tone is a class suffix rather than `data-tone`
(`.skills-status-*`, `.skills-badge-*`, `.toast-*`, `.marketplace-state-*`,
`.chat-live-phase`, `.log-phase`, `.evo-runtime-pill`, `.widget-table-status`,
`.widget-metric` / `.widget-callout`) name the same tokens. A surface that
paints only its edge (a card tinted by its state, a callout's left rule) takes
the border and leaves its own background alone; a toast keeps its glass
background, because a translucent status fill over live page content costs the
text its contrast.

Adopting these tokens is applying the semantic status contract, which already
governs every surface — it is not a token migration of those surfaces and does
not move them into the migrated set in section 8.

## 5. Card and section composition

- A panel is one `.ui-card`-family surface: `--ui-card-border`,
  `--ui-card-bg`, `--radius`. Nested emphasis uses `--ui-card-bg-soft`, not a
  second border weight.
- A section is: title (`--type-section`) → optional one-paragraph description
  (`--type-body`, `--text-meta`) → content → optional note (`--type-meta`,
  `--text-meta`). The description explains what the section decides; the note
  carries consequences and caveats.
- Subsections inside a section use a `--type-body` semibold heading and stay
  visually grouped with their own rows, their own add action in the head
  (List editors, below). A heading that floats equidistant between two groups
  belongs to neither.
- **A collapsed disclosure shows that it opens.** A `<details>` summary always
  carries a visible open/closed marker — the native triangle, the `▸`/`▾`
  glyph pair, or button/card chrome. A summary that sits beside a help line at
  the same size takes control ink (`--text-primary`) and its own line, so it
  does not read as one more note; a summary that already reads as a control
  through its own chrome or placement may stay in meta ink. Overriding
  `display` on a summary drops the native marker, so the glyph must be drawn
  explicitly.
- Spacing comes from the 8pt tokens (`--space-*`); a new visual dimension
  becomes a CSS variable before it becomes a page-local literal.
- An item in a popup menu or a picker list highlights with
  `--menu-item-hover`. One gesture, one fill: a menu that highlights at a
  different strength than the menu beside it reads as a different control.
- **Content text is always selectable and copyable.** A control may suppress
  selection (`user-select: none`) only on its own label or chrome, never on
  content it contains. Where content lives inside a click-to-toggle surface
  (a task card's summary), the surface ignores a pointer click whose drag
  produced a non-empty selection; keyboard activation is unaffected.
- **A markdown heading inside chat is a subsection label**, never a page
  title: in chat bubbles every heading level renders at `--type-body`
  semibold; in a task card's timeline it renders inline, without block
  margins, at its row's own size, with a copyable line break before the
  following paragraph. The page-size `md-h1` belongs to non-chat
  surfaces only.
- **A task card's summary outranks its details.** The latest-activity line is
  `--type-body`; collapsed timeline rows are a dense log at `--type-meta` in
  `--text-secondary`; an expanded row returns to `--type-body` in
  `--text-primary`. Details never render larger than the summary above them;
  an inline label inside a row is semibold at the row's own size.
- **A nested child card is subordinate to its root.** Its compact identity
  row shows status, role, notes and a chevron; a short task id disambiguates
  otherwise identical siblings. Executor facts and the agent/coordinator model
  occupy the metadata row, so the task's coordinating model cannot masquerade
  as its external executor. A child keeps one useful activity line visible;
  a root permits up to three. Empty activity reserves no band, and a duplicate
  title is not activity. Full narration and Reviews expand independently.
  The root keeps primary title ink at weight 500, children secondary ink at
  400. Nested frames preserve real ancestry; their opaque secondary surface
  avoids accumulating translucent white tints at greater depth.
- **An executor label states its evidence.** A progress actor is labelled
  `last update`, with requested model or `model unconfirmed`; it is not proof
  of current computation. Settled observed models are separate historical
  facts, not a claim that their union is the current actor. Missing identity
  stays unconfirmed; marks and configured routes never manufacture execution.

### History edges

A paged transcript loads older portions automatically at the reading edge and
keeps a keyboard-reachable `Load older messages` button that retries the same
portion when reading fails. A short or empty portion never claims the beginning
of the archive; only the source reader establishes that boundary, and an empty
portion is never a reading position. Distant portions may leave the rendered
window and return quietly as the reader nears the live edge. There is no
`load newer` control: the one explicit return to the present is the floating
`Scroll to latest message` button. An edge control states a fact about the
rendered transcript, never about an internal cache or cursor. The visible
passage, selected text, focused control and expanded Reviews retain their
actual nodes.

### Project work pointer

A Project keeps its conversation and real nested task cards. One compact pointer
leads to an unfinished represented root, or the latest represented root when all
are finished. It occupies one line: it names the card (its coined name, else its
title) and ellipsizes rather than restating a status headline in full, so the status bar
never grows into the reading area; the complete text stays on the card itself,
one click away, not in a mouse-only tooltip. A default desktop panel keeps the
pointer, the coverage note and the status pill on one row while the pill is
short (Online, Working, Thinking, Sending, Queued); a longer pill, a narrower
panel or a phone wraps the bar to a second row, never a third. It states `Loaded messages only`
unless history coverage is complete; without a represented card the pointer and
that note are hidden, which is not a claim that the Project has no work.
Navigation moves the conversation to the existing card without changing the next
message's recipient, opening another work pane or manufacturing activity.

### List editors

A list editor is any section where the owner adds and edits entries in place:
the Available subagents roster, the Review lanes groups, MCP servers, custom
keys.

- A section-level add action acts from its group's header (§6). A list
  editor's new entry appears at the end of its own group, is scrolled into
  view — the shortest distance, without animation — and takes the caret in its
  first field. A button that stays in view while the entry it made is born
  off-screen has not finished its job.
- A freshly added entry is an invitation, not an error. Where a list editor
  validates in the browser (today the Available subagents roster), the entry
  shows a neutral hint in its own meta line until the owner tries to save; the
  error then names the entry and stands beside it — the entry tinted with the
  status pair, never dimmed — with the section-level line as the summary. A
  save attempt judges the entries that existed then; one added afterwards is
  an invitation again.
- A multi-field card (an MCP server) follows the add-and-reveal rule without
  adopting the §6 row anatomy.

### Reviews inside task cards

Real tasks and real subagents are cards. Reviews are a subsection of the
exact real task that owns their presentation. Harness and neutral API marks
identify the delivery channel alongside explicit execution evidence; they are
not child-task cards and never prove execution by themselves.

- A collapsed task card shows only a quiet `Reviews N` count, docked on the
  metadata row (it wraps under the metadata on a narrow card), optionally with an
  active count; a collapsed nested child card docks it on its metadata row the
  same way. It has no aggregate pass/fail alert, no synthesized verdict, and
  no review dollars.
- Expanding `Reviews` reveals one row per currently admitted review group
  (`Skill review`, `Plan review`, or `Task acceptance`). Expanding a group
  reveals its ordered attempt rows. Group state and verdict remain
  domain-specific; one blocker never recolours the whole task card.
- Start progress labels the frozen model/route/profile as requested; settlement
  reports that same slot's observed execution or says it was not reported. An API
  model sent in a request is not an independently observed provider label, and
  duplicate model slots remain distinct. No global last-run identity fills a gap.
- Disclosure is user-owned. Review results, retries, failures, terminal task
  state, reconnect, and lazy-detail loading update content in place but never
  open or close the task, Reviews section, or group.
- A panel that settled after its task ended stays one attempt row of its group,
  labelled as settled after the task ended; its note (which verdict, which
  revision, whether a reviewer's outcome is still unknown) is host-composed and
  printed verbatim, leading the attempt detail.
- Stable keyed rows are reconciled in place. A routine update preserves the
  exact lazy-detail node, focused descendant, and its reading position. Expanded
  groups state exact aggregate accounting when projected and otherwise say
  `Cost unavailable`; attempt detail states exact accounting when the domain can
  prove it and otherwise says `Cost unavailable`. Collapsed rows never show dollars.
- Harness marks are monochrome `currentColor` vectors with adjacent visible
  text. They carry identity only, remain neutral across status states, and use a
  generic text-preserving fallback for unknown harnesses; direct API is shown
  neutrally as `API`.
- Vector provenance: Claude, Cursor, and OpenCode paths come from Simple Icons;
  Codex/OpenAI comes from SVGL. Product names and marks remain the property of
  their owners.

### Quiz card

The owner quiz card (`web/modules/chat_decision.js`, `.chat-quiz-*` in
`web/style.css`) is a chat-delivered decision surface. Optional clarification is fire-and-continue: the task states its assumption.
Required waiting keeps the same card and explicitly says that the task awaits
the owner, with Stop and existing task deadlines still effective; an optional
bound on that wait resumes the task with a host notice and leaves the card open,
and that notice says the same thing the card does — with a stated assumption the
task proceeds under it, and without one, no answer is explicitly not consent.
Silence is never an answer on either surface. A card outlives its asking task:
after the task finishes, the owner can still answer, and the answer arrives as
their own message in that chat. After settlement, status and the owner's recorded
answer keep both forms readable. Anatomy, top to bottom:

1. **Head** — neutral `Question` chip (`--type-meta`, neutral pair) and a
   status as dot + text. The lifecycle word family is closed and leads with the
   one word that answers "is there an unanswered question for me?":
   `Waiting for your answer` needs positive wait evidence (the task's live wait
   record, or the original required flag before any record exists); a resumed
   wait — the owner typed instead, or the bound closed — reads `Unanswered · the
   task continued; an answer is still accepted`; an open question without any
   wait evidence reads `Unanswered · an answer is still accepted`;
   `Unanswered · the task finished; a late answer is accepted as your message`
   keeps the neutral dot and answerability; `You answered` uses the ok dot;
   `Replaced by a newer question` uses the disabled dot; an unreadable source
   reads `Status unavailable`, never an invented invitation. No answer-deadline
   countdown: task completion closes its mailbox, not the question's answerability.
2. **Question** — the one primary thing: `--type-body` semibold,
   `--text-primary`.
3. **Stake** — optional one-liner (`At stake: …`), `--type-meta`, `--text-meta`.
4. **Options** — real owner actions: buttons with `--text-primary` labels,
   legible at rest; an optional per-option detail steps down to meta ink.
   After settlement buttons drop to `--text-disabled`; the chosen option keeps
   the ok pair. Options are capped by the shared Python↔JS constant
   (`MAX_QUIZ_OPTIONS`).
5. **Free answer** — while the card is open, a compact always-visible field
   (`Your answer or comment…`) with a `Send my answer` button, enabled only
   once something is typed. No option ever has to be the least wrong one: the
   text rides with an option click as the owner's remark, or goes alone as the
   owner's own answer. It uses the card's own ink and surface tokens (never
   the legacy chat input), is capped by the shared Python↔JS constant
   (`MAX_DECISION_COMMENT`), and disappears the moment the card settles.
   A settled card instead carries what the owner said as a second primary
   line (`Owner's answer: …`, `--type-body`, `--text-primary`) under the
   options — beside the highlighted option when one was chosen, and as the
   whole answer when none was.
6. **Assumption or waiting** — the signature line (`Continuing meanwhile: …`
   for optional clarification, an explicit waiting statement for required input),
   `--type-meta`, `--text-meta`, separated by a hairline. While the card is
   open it names the default path. The optional assumption remains after settlement
   as the record of work continued meanwhile. Required waiting copy disappears
   when the card settles; the status and recorded owner answer remain.

The card was born on tokens ahead of the rest of the chat surface (which has
since migrated too): type sizes and every colour come from tokens (no new
literals), the chip's pill radius and the option gap included; every focusable
element in the card shares one keyboard ring (2px `--focus-accent-border`,
2px offset). Component geometry (card min/max width) keeps local literals like
the rest of the chat surface.

**Project question mirror.** A Project question the owner has not answered appears in Main as the Project's own quiz card — the same `buildQuizCard` form with the question through the chat markdown pipeline, the options with their details and the `recommended` badge, the stake, the assumption or waiting line, the status and the own-answer field — inside the same assistant bubble. The one addition is a Project chip in the head beside the `Question` chip: a pill in the project chip's own language (the `--project` tints, the Project name in project ink with `↗`) that opens that exact question in its Project, with the card's shared keyboard ring. A long Project name yields first (the chip is capped and ellipsized, its title names the Project whole) so the status keeps its place; a phone column wraps the head. Every lifecycle state reads as it does in the Project: waiting, open, resumed and finished questions stay answerable, and a replaced question stays as a read-only record. An unreadable source keeps what Main already knew; with nothing known the copy says `Status unavailable`, takes no answer and keeps its chip, and a row that cannot carry the form yet shows `Open the original question for its text.` until it can. The first confirmed answer from any source — a press in Main, the Project form or another device, a history or census snapshot — shows the recorded result (the chosen option, `Owner's answer: …`, `You answered`) for five seconds and then removes only the Main copy, through the ordinary message retirement and without moving the reader's viewport; the Project keeps its card. The countdown starts once and later observations never restart it. When focus was inside the copy it stays there while the result shows, then moves to the next Main question, or to the composer for a keyboard owner, never summoning a touch keyboard. A copy that learns its form and its answer in one delivery shows that result for the same five seconds. An answered question never enters Main again: fresh history, a reconnect or a stale open snapshot cannot bring the copy back. Main remembers the lifecycle of a bounded number of questions; a question it no longer remembers mounts a safe unknown copy, whose answer controls appear only after a fresh canonical record confirms it unanswered. A failed, missing or wrong-project canonical read leaves a safe `Status unavailable` copy with its Project chip and no answer controls; the chip opens the original Project form, while a later owned refresh retries the Main copy, so an unavailable read never turns a stale open snapshot into an answerable form and never permanently suppresses a legitimate unanswered question. The mirror and the quiz header share the lifecycle wording above.

Project-lifecycle and routing actions use the shared `createSystemMessageActions` composition. It owns token-based space above and below the controls, wrapping and clearance for the existing button focus ring; action buttons never sit in a clipped/nowrap text line. This is a row composition, not a new card framework or a global button-margin rule.

History with no current execution or known outcome keeps its expandable content under `Outcome unavailable`, without a task chip, typing or Stop. Before complete live-source reconciliation, it is `Activity unconfirmed`. Positive current activity restores only its proven controls. A delivery warning may coexist with a preserved task-acceptance PASS. Model metadata says `Last solve response`, naming the initial request only when the route changed.

### Conversation activity block

A task's activity block is in the transcript exactly when the record already
holds something to show — one predicate (`web/modules/chat.js::blockVisible`),
re-read at every mutation, with no sticky flag: no kind is shown
unconditionally (a Presence turn and a consciousness wake-up are direct turns
and follow the same rules as any other); open owner attention (a model wait, a
pending stop, a host-attested Stop the record still offers — the same reading
the control uses, so a block never stands on a Stop it hides); a child card; a
review group; a content row; a terminal outcome other than Done. The completion note is not content:
a turn that ran no tool and finished Done leaves no block — live, after a
reload and after a reconnect.

Narration leads; routine execution evidence stays compact; exceptions keep
their explanation and controls. The block's title, its collapsed activity line
and its timeline carry the turn's own narration — the progress frames the
model itself authored, marked by the typed `narration` fact the worker stamps
at that one producer (a frame without the fact is a legacy frame and is read
as narration) — and whatever needs the owner's eyes: a failed or timed-out
step, a wait, a review, a child. What the host says about how the turn is
running (a checkpoint, a model fallback, a review verdict, a nudge) is a
visible timeline row that never claims the title or the collapsed line, so a
turn whose only notes were the host's keeps its coined or task name and an
empty activity line. Successful tool calls are not rows at all: they fold into
ONE evidence row per block — `N tool calls`, or `N tool calls · M errors` once
a call failed — that stands at the first call's position and time and is
patched in place; Expand shows the per-tool counts (`read_file ×3 ·
web_search`); its phase is `calling` while a tracked call is still running,
`warn` once a call failed, `result` otherwise, and the row says that phase in
ink rather than in extra words. A failed or timed-out call keeps
its own error row (content) and is counted in the evidence total.
`web/modules/chat_activity.js::toolEvidenceView` builds that row for the live
path and for the recorded metrics alike, so at rest the row carries the same
counts and names live, on reload and on reconnect; a cold reload mints it from
the metrics, so it carries the metrics' time and sits where the metrics
arrived, while a reconnect keeps the live position. Live it derives from the
observed call frames (once per call identity; identical repeats without an id
collapse into one), and the host's metrics replace those numbers as they
arrive, field by field: a fact that states a total says nothing about the
routing or error count, so it can neither erase one nor reclassify a receipt
row into content, and a call frame after the terminal changes nothing. Block presence is the same live, on
reload and on reconnect (a turn that moved itself into a Project with
`ensure_project_scope` is the exception: its block and answer live in the
Project room, and Main replays only the owner message and the Started
annotation); a child card reads the same voice rule for its own notes and folds
its calls live, but replays no evidence row.
`N notes` in the collapsed header counts timeline items, the evidence row
among them.

The block's chrome follows the work it stands on
(`web/modules/chat.js::blockHasWork`, the presence facts minus open attention
and minus a bare terminal outcome), never the lane that ran the turn (owner
decision 16.09: real work is a task card, a greeting is nothing). A block with
work — a review group, a child card, an evidence or narration row, a tool
error — is the task card whether a managed root or a direct
conversation turn produced it: a title (the coined name, the latest narration
headline, or the `Working…`/`Task activity` placeholder), the status chip, Stop
while the host attests it, and `Turn into project` in Main unless its origin is
already bound (a direct turn's later rows then route to the Project room like a
turn that called `ensure_project_scope`). A nested child card is work inside
its root's block, never a block of its own: it carries neither control, and its
root's conversion is the conversion of the whole work. A block that exists only for open
attention — a model wait, a pending or host-offered Stop — or only for a
non-Done ending of a turn that did no work carries no title placeholder and no
conversion; its chip says the state it is in (Waiting…, Cancelling…, Failed),
the wait controls stay, and its first row of work gives it the title. The
collapsed header carries the tool count live and, once the turn ends, cost and
duration (a replayed header carries the count and cost; duration is a live
fact); its `updated` stamp follows the turn's own narration, never a host note
and never a tool call. The
host's `_is_direct_chat` fact keeps its host jobs (routing, census `kind`, Stop
custody, terminal rows) and, on the client, only the header pill (a direct turn
keeps the census verdict beside its block). A block whose only reason to exist
was open attention leaves when that attention closes: a wait-only block
disappears when the wait resolves, and the resolved episode's no-reopen ledger
survives with the record, so a stale revision cannot bring the block back.

An addressing call (`promote_chat_to_task`, `route_to_project`, `steer_task`, `ensure_project_scope` —
the routing-verb family `ouroboros/tool_capabilities.py::ROUTING_VERBS` owns) is stamped by
the host on its live tool-call frames (`routing_action`) and counted in the
task metrics (`routing_tool_calls`); its receipt is the typed routing
annotation on the owner's message. Such a call is a receipt row: it renders
inside a block that exists for other reasons but is never content the block
stands on, and the evidence row of a turn whose calls were all addressing
calls, without error, is a receipt row too — live from the stamped frames, on
reload from `routing_tool_calls`. So a turn that only
addressed work («turn this into a project») draws no block, live or on reload:
the annotation on the owner message and the managed root's own card are its
whole record (owner decision 11.09). A failed addressing call is an error row
and therefore content, as is any recorded tool error. A REFUSED addressing act
is told where the work lives and never in Ouroboros's voice (owner 16.09): the
receipt line states the cause in the owner's words, the failed call stays the
error row inside the block, and the tool result carries the typed reason (with
the cause and repair in `detail` where the producer holds one) so the model
narrates — no host bubble interrupts a narrating
turn. When the host itself issued the act (a Swarm message, a skill-card repair,
a picker click), or a refused owner steer carries no owner-message receipt,
the host states the refusal as ONE typed System row: `task_not_started`,
`task_start_unconfirmed` for unconfirmed admission, or `steer_not_delivered`
for steering. It stays in the issuing chat, keyed to the named target. The
Project start row is announced only once the task is really queued. A refusal receipt with neither options nor a cause sentence
reads «Not routed», never «Choose a target». No client list of tool
names decides presence (`docs/development/02-naming-and-boundaries.md`, "an
open default behind a closed exception list"), and no client reading of a
note's text decides whether it is narration. A change that draws one row or
line per event is judged at a realistic burst size — a multi-call turn,
collapsed and expanded, at desktop and phone width — never at a two-event
fixture.

### Subscription waits inside task cards

Quota exhaustion and a confirmed need to sign in again use the same component,
`model_wait.js` with `model_wait.css`, inside the turn's existing host: the task
card of a turn that has done work, the bare block of a turn that has not. Each
waiting role has its own row; the model, account and reason are separate facts.
The controls stay visible when the task's timeline is collapsed. Waiting carries
a quiet warning status and no computation animation, activity counter or invented
progress. Completed steps remain intact; a held worker slot and its effect on
the queue are stated only when the host reports that fact.

The quota row offers automatic continuation, initially enabled, and shows a
known reset time or an explicit unknown. An authentication row instead offers
the existing Accounts sign-in flow. Both can open Settings, retry explicitly,
or use the shared model-role editor to choose a replacement model/account.
The replacement affects the named waiting role until the task ends; an unchecked
"Also save this role in Settings" checkbox separately requests persistence.
Fallback Local remains shared by its Settings group. Changing Local for one
waiting fallback makes the replacement task-only: the persistence checkbox is
cleared and disabled with an explanation, while Apply remains available. Model
and account changes can still be saved when the shared Local value is unchanged.
Permanent group-wide Local changes remain in Models.

A pool confirmed to contain both accounts requiring sign-in and accounts waiting
for quota says "Waiting for access" and names both causes. It keeps automatic
continuation and opens Accounts without selecting a profile or initiating login.
Unknown pool availability is not this state. The confirmed quota component uses
the same execution-clock pause; an authentication-only wait still consumes that
clock, and calendar deadlines stay fixed.

A submitted action is shown as pending until the task reports its application.
A saved Settings change and a still-pending task change are disclosed separately.
Retries preserve the original request identity and payload. Revisioned rows reject
older observations, resolved episodes never reopen, and a terminal task removes
all wait actions. A mailbox delivery failure keeps Retry request available for
that same accepted command, even while its application remains pending. The
current task attempt selects live actions; a previous attempt's retained pause
never makes a new working attempt appear to wait. Updates and history rebuilds preserve the same keyed editor,
its draft, persistence checkbox, focus and reading position. Wait pickers omit
Context controls on every render: their action changes model/account only.
Waiting chips do not pulse or show typing. Closing the chat disposes view
resources without claiming that it stopped the task; continuation requires the
Ouroboros process to remain running.

A consciousness wake-up needs nothing of its own here. It is an ordinary direct
turn with its own task id, so its model wait, its controls and its temporary
model change behave exactly as an owner turn's and last until that turn ends;
the persistence checkbox still saves the consciousness role. What identifies the
turn is the origin label `Consciousness` in the block's meta line, on its final
bubble and on the cards of tasks it started — never a separate card, a reused
slot or a different vocabulary.

An already-delivered answer does not close a still-open post-task synthesis.
Reflection or consolidation waits use the same role controls in that task's
existing finalizing card. Pooled tasks, including API-only tasks, keep their
worker slot until post-work settles; the answer arrives early. Ordinary native
chat post-work holds no worker slot; its existing card keeps the same live
post-task model-wait controls after ordinary dialogue admission closes. Both claims follow the host's live owner
and post-task checkpoint, not the presence of answer text or a cost estimate.
Failed main work stays visibly failed after history reload while post-work
controls remain live; the unfinished checkpoint never erases the outcome.

## 6. Account group / row anatomy

For a repeated identity row (a connected agent account, a reviewer slot,
a server entry):

1. **Classification chip** — neutral pair, `--type-meta`. Only where the row's
   family is not already expressed by the group it sits in; inside a per-family
   card the chip repeats the header and is dropped.
2. **Name** — `--type-body` semibold, `--text-primary`. The one primary thing.
3. **Identity detail** (email, plan) — `--type-meta`, `--text-meta`.
4. **Status** — dot + text from the status pairs.
5. **Meta line** — `--type-meta`, `--text-meta`, on its own line under the
   name. Quantities are stated in human words ("38% used · resets in 2h"), and
   an instant is humanized. A row never leads with a raw ISO timestamp.
6. **Actions** — docked right, legible at rest. A control rendered at
   secondary ink reads as disabled; if the owner can click it, it is
   `--text-primary`.

For a row with one action and a durable result, the result occupies the flexible
left side and the neutral action stays docked on the right. Field-level actions
(for example Show/Clear) keep the field's control height; they are not reused as
the compact result-row action.

Rows of the same kind are equivalent: no row gets extra visual weight for
being first, default, or native. Grouping and section-level actions express
which family a row belongs to, and a section-level action (add, connect)
belongs in its group's header rather than attached to one privileged row.

**A degraded row is emphasised, not dimmed.** Lowering a whole row's opacity
greys out the sentence that reports the problem and makes its still-clickable
controls read as disabled. Tint the row with the matching `--status-*-bg`
instead, and let the status text carry the claim.

## 7. Onboarding density

Accounts is the common connection surface for subscriptions and API keys. Open sign-in link hands off to the current desktop, browser or Telegram host while retaining the wizard; Copy is separate. An unavailable host opener reports a retryable failure, and a supported copy fallback says that it copied rather than claiming an open.
Models and Agents edit assignments; adding a connection updates available
choices without replacing an owner's assignments. A model role uses one compact
Source / Model / Account row. The account is a property of that role: Auto
rotates compatible accounts and an explicit pin stays pinned. A model inherited
from Main remains visibly inherited while its account can be pinned
independently. Fallbacks use the same row in their saved order, with adjacent
move/remove controls and the group's Add action.

Context details distinguish the exact route's advertised Auto window from a
manual value labelled "set by you". Changing an account withdraws the previous
account's metadata immediately, including during a failed or pending catalog
read. Unknown limits stay unknown. Catalog updates keep the edited field and
caret in place and never assign a model. `model_roles.js` and `model_roles.css`
own the shared Settings/wizard editor; `reviewer_slots.css` supplies the same
reviewer-row layout to both documents.

A source is chosen, never spelled. Every surface that assigns a model — the
Models roles, Available subagents, every review lane, the first-run wizard and
the quota-wait picker — offers one grouped source select with the same groups
in the same order: configured subagents where references are allowed,
Subscriptions · models, API keys (one entry per provider with a stored
credential, then one disabled pointer to Accounts; a saved choice without a
credential stays selectable as "(no key)"), Agents · sessions where a session
is possible. The model chooser lists only the chosen source's catalog, so a
suggestion's transport is the selected source; any id can still be typed. The
stored spellings (`provider::model`, `claudexor::source=model`,
`harness=model`) are serialization authored by the editor: never required from
the owner, never a field placeholder or help-text instruction, never the
primary displayed value; the exact stored id may appear in a meta line or
tooltip. The route identity chip names the source (API · OpenAI, Codex · model,
Claude Code · agent), not the channel alone. A last-run receipt is shown
against the route that produced it: when the row's route changed since, the
line says so and names the earlier route.

Subscription model sources and agent sessions never imply each other's model
inventory. Source ids are opaque; the model-sources catalog names the
credential harness. Saved subscription-model account pins survive catalog gaps
and unrelated saves; direct API-key models do not offer a subscription-account
pin. Catalog entries are suggestions, not account-specific entitlement or
context evidence. Changing a model or account never changes the delivery kind:
a configured subagent reference remains a reference to its native inspection
episode, a scope or deep self-review row keeps reading the repository itself
whatever model it names, and an inline packet row stays inline. Catalog refreshes
preserve the edited value, focus, selection and scroll position. Returning to a
reviewer's previous source restores that source's model/account draft; a source
not previously selected starts without another source's pin.

The wizard has five steps: Accounts, Models, Review, Budget, Summary. Agent
connection is inside Accounts; Codex is the recommended connection for starting
without an API key. Other existing agent connections describe their actual agent
capability. Connected reports sign-in, independently of the model-source and
suggestion reads. Accounts names pending, failed or partial reads and offers a
contextual Retry that preserves the current fields. A known model source allows
Continue and manual model entry even when its inventory or automatic suggestions
cannot be read; an unknown source explains why Continue is unavailable. New
subscription-only installs clear only untouched shipped API suggestions without
access, requiring Main while Light can inherit it and Fallback can stay empty.
Stored or edited values remain intact.

Review & start computes the skipped model and reviewer steps before showing
Summary. After a failed automatic setup, an explicit recovery action prepares
all reviewers on the selected Main model, including its account and processing
choice. This is disclosed as one model for every review, not model diversity.
The resulting Summary is shown before a separate Start saves it; subsequent
manual edits remain authoritative. Completion without the automatic preset
leaves later configuration to Settings. Summary names exactly the assignments
the one atomic Finish saves, including deep self-review. Reviewers remain
editable with the same controls as Settings. A subscription-only Budget step leads with quota/reset facts and keeps
optional API spending fields collapsed. "No API key" never claims unlimited free
work or that paid provider credits were enabled.

The first-run wizard is a compact flow that must not scroll at the default
desktop window size merely because a step has several fields.

- Step title `--type-page`; card titles `--type-section`; field labels and
  notes `--type-meta`. No display size above `--type-page`.
- Short-viewport adaptation hides explanatory copy rather than shrinking type.
  Once copy is hidden, shaving pixels off a title buys nothing and costs the
  scale.
- Field labels are sentence case at meta ink — a wizard step shows a dozen at
  once, and its job is to get one value typed, not to present a grid of
  headings.

## 8. Migration state

The scale is applied to complete component families and declared page regions.
Using a migrated field inside a historical page does not claim the whole page
has migrated. Migrated today:

- `web/ui.css` (shared palette, fields, buttons, status/chip recipes and
  menu/chooser chrome, used by both top-level documents and optional author pages)
- `web/settings.css` (settings shell, model/effort cards, MCP cards)
- `web/onboarding.css` (the whole first-run wizard)
- `web/model_roles.css` and `web/reviewer_slots.css` (shared role editors)
- `web/style.css` between the `design-system:migrated-begin` and
  `design-system:migrated-end` marker pairs (several — migrated surfaces are
  not contiguous in the file): harness accounts, reviewer slots, the
  Dashboard → Updates tab (status card, one action row, collapsed Recovery
  with a single restore list), and chat (typography, foreground and status
  colour; component geometry keeps its local literals per the viewport
  reserve contract, while the shared palette channels keep translucent glass
  surfaces coherent across themes)
- the global `.muted`, `.form-section h3` and shared `.ui-status` tone rules

The remaining page-specific typography in skills, marketplace, widgets, logs
and evolution is historical. Their adopted common controls follow the shared
family; unrelated page rules keep their literals until their own pass. Migrate
each selected family completely and remove its replaced recipes in the same
change. Do not introduce a half-tokenised second field family or describe a
control adoption as an all-page redesign. The semantic status/action/notification
contract already applies everywhere.

`tests/test_web_typography_static.py` guards the migrated set only. Extending
the guard and migrating the corresponding family or region are the same commit.

### Author freedom

An extension may use the optional shared buttons, fields and status functions
inside its own module or route-iframe page, override them, or design a completely
independent interface. `.ouro-ui` supplies font and native dark-control context;
the named classes opt controls into the recipes, with no page-wide reset.
The kit reads the installed source at a new mount; retained frames keep the styling
they loaded. It introduces no theme polling, forced remount or mandatory visual
conformance. Author layout, validation, operations and loading feedback remain
author-owned; the small source recipes are in `docs/examples/author_ui_kit/`.

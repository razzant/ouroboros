# Ouroboros Design System

Normative authority for the **visual and interaction semantics** of the Ouroboros
UI: what a size means, what a colour claims, what a chip is allowed to say.

Authority split:

- **This file** decides semantics — type scale, hierarchy, foreground and state
  colour, status conventions, card/row anatomy, density.
- **`docs/DEVELOPMENT.md` → `## Design System`** decides the engineering rules
  that keep those semantics intact — where values may live, which component is
  the SSOT, what counts as review debt, how a visual change is verified.

Values themselves live in `web/style.css` `:root` (and are mirrored by value in
`web/onboarding.css`, which is inlined standalone and cannot import it). This
file names roles; it does not copy an inventory.

The theme is **dark only**. There is no light-theme plumbing, and adding a
second theme is an architecture change, not a styling change.

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

The first-run wizard carried a *second* brand red for a while, so the first
screen a new owner saw was the one screen that did not match the app.
`web/onboarding.css` is inlined standalone and cannot import `web/style.css`,
so it mirrors the shared tokens **by value**, and
`tests/test_web_typography_static.py` fails if a name declared in both `:root`
blocks resolves differently. A wizard-only token is fine; a wizard-only *value*
for a shared name is not.

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

## 4. Status and chips

A status has **an explicit foreground/background pair**, never a foreground
derived from whatever generic opacity happens to sit on the element.

Status, owner action, and urgent notification are separate product concepts:

- **Status** states a fact about the affected object. It does not imply that the
  owner can or must act. Task status uses one factual word family: `Working`,
  `Done`, `Done with warnings`, `Failed`, `Cancelled`.
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

Local diagnostic failures remain inspectable in details and Logs, but do not
relabel the whole still-working task. A failed child keeps a compact factual
`Failed` marker inside its parent while the root continues under its own
authoritative status. Internal reason codes belong in details and diagnostics,
not compact headlines.

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
  visually grouped with their own rows and their own action toolbar. A heading
  that floats equidistant between two groups belongs to neither.
- Spacing comes from the 8pt tokens (`--space-*`); a new visual dimension
  becomes a CSS variable before it becomes a page-local literal.
- An item in a popup menu or a picker list highlights with
  `--menu-item-hover`. One gesture, one fill: a menu that highlights at a
  different strength than the menu beside it reads as a different control.

### Reviews inside task cards

Real tasks and real subagents are cards. Reviews are a subsection of the
exact real task that owns their presentation. Harness and neutral API marks
identify the delivery channel alongside explicit execution evidence; they are
not child-task cards and never prove execution by themselves.

- A collapsed task card shows only a quiet `Reviews N` line, optionally with an
  active count. It has no aggregate pass/fail alert, no synthesized verdict, and
  no review dollars.
- Expanding `Reviews` reveals one row per currently admitted review group
  (`Skill review`, `Plan review`, or `Task acceptance`). Expanding a group
  reveals its ordered attempt rows. Group state and verdict remain
  domain-specific; one blocker never recolours the whole task card.
- Disclosure is user-owned. Review results, retries, failures, terminal task
  state, reconnect, and lazy-detail loading update content in place but never
  open or close the task, Reviews section, or group.
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
`web/style.css`) is a chat-delivered decision surface. It is fire-and-continue
UI: the asking task keeps working under a stated assumption, so the card must
read correctly both as an invitation ("you can redirect me") and, after it
settles, as a record of the path taken. Anatomy, top to bottom:

1. **Head** — neutral `Question` chip (`--type-meta`, neutral pair) and a
   status as dot + text. The lifecycle word family is closed:
   `Awaiting answer` (neutral dot), `Answered` (ok dot),
   `Task finished — question expired` (disabled dot), `Superseded by a retry`
   (disabled dot); an unknown state keeps a neutral dot but reads as settled
   `Closed`, never as an open invitation. No timers, no countdowns: a quiz expires only with its
   asking task.
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
6. **Assumption** — the signature line (`Continuing meanwhile: …`),
   `--type-meta`, `--text-meta`, separated by a hairline. While the card is
   open it names the default path; once the card settles it is the durable
   record of what the agent did without an answer. It is never dropped on
   state change.

The card was born on tokens ahead of the rest of the chat surface (which has
since migrated too): type sizes and every colour come from tokens (no new
literals), the chip's pill radius and the option gap included; every focusable
element in the card shares one keyboard ring (2px `--focus-accent-border`,
2px offset). Component geometry (card min/max width) keeps local literals like
the rest of the chat surface.

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

The scale is applied surface by surface. Migrated today:

- `web/settings.css` (settings shell, model/effort cards, MCP cards)
- `web/onboarding.css` (the whole first-run wizard)
- `web/style.css` between the `design-system:migrated-begin` and
  `design-system:migrated-end` marker pairs (several — migrated surfaces are
  not contiguous in the file): harness accounts, reviewer slots, the
  Dashboard → Updates tab (status card, one action row, collapsed Recovery
  with a single restore list), and chat (typography, foreground and status
  colour; component geometry keeps its local literals per the viewport
  reserve contract, and the glass surface tints — frosted header/composer
  backgrounds, bubble gradients and their border tints — remain local
  literals with no token equivalents yet)
- the global `.muted`, `.form-section h3` and shared `.ui-status` tone rules

Not yet migrated: skills, marketplace, widgets, logs, evolution. They are
historical and keep their literals until their own pass. Do not part-migrate a
surface: a half-tokenised stylesheet is harder to reason about than an untouched
one. The semantic status/action/notification contract above already applies to
these surfaces; it does not by itself authorize a visual token migration.

`tests/test_web_typography_static.py` guards the migrated set only. Extending
the guard to a new surface and migrating that surface are the same commit.

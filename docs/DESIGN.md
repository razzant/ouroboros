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
composition, not a second shared palette. This file names roles, not an inventory.

## Design bible: "Airy dopamine"

This distils the interface Ouroboros already has into rules, so every agent extends
it the same way. Every rule below derives from these lines and holds in both themes;
a component that needs its own exception is a design defect. A rule names a role (whose
object, what is chosen, what is primary); how a theme draws that role is its seed below,
so a recipe is written once for both themes and a theme differs only in its token
values (the few older light-only rules in style.css stay as they are and are not a
pattern to copy).

**In one sentence (for a designer).** A light, borderless soft UI in two themes: a
plain room (white, or deep graphite with glass where controls float over content); voice-coloured objects, each one
flat fill resting on a soft shadow; everything pressable a capsule (a card when it carries a paragraph) standing on that
shadow in one of three tones, whose states change its colour, never its height, as on macOS; one crimson accent and
candy-coloured status words and marks; Apple system type in four sizes. iOS airiness with Material 3 tonal colour; no
gradient behind content or on an object, and no glows, textures, outlines or neumorphism.

**Principles**

1. **The room is plain.** Page, sidebar and panel backgrounds are the room colour or
   one step off it. Never a gradient, glow, wash, texture or animation behind content.
2. **Objects rest; they are not outlined.** A bubble, card or group is one flat fill
   on the *rest* shadow: no gradient on it and no light along its edge. What floats over moving content (the Shade, menus, the
   composer field, dialogs, the drawer, ↓ latest) takes the *float* shadow. Inside an object a
   group or row is a *tray*, a soft neutral fill without a shadow, and fills never
   stack. Only fields keep a hairline, because an editable box must read as editable;
   the composer's field floats instead, and focus gives it the field's ring. On a dark
   room a shadow barely shows, so each neutral height is also a lighter surface: a white
   veil over whatever lies under it (room, object, capsule). Room capsules use the neutral veil. Inside the dark
   Ouroboros voice, trays and capsules use opaque lighter steps of that voice, with opaque readable ink. What floats
   over moving content is glass. The Shade is the one exception: it
   continues its bar's ink and hangs on the float shadow, so the bar and its Shade read as
   one surface unfolding.
3. **Colour says who and what.** Each voice colours its own objects: the owner blue,
   Ouroboros rose (its answers and its work cards alike), System amber (gold on dark), a
   Project fuchsia. A role keeps its hue family in both themes and the theme gives it its
   material: on white a pale
   tint of the hue, on a dark room a deep opaque colour of it, because a tint of the same
   hue over black turns brown or ash. Yellow, amber and orange turn olive or brown whenever
   they fill a dark surface, so there they never fill: they are the name and the
   word's ink on a neutral body (System, the owner's turn, money, a warning), and a state's
   green or red draws the same way on dark: a row whose light body is tinted keeps a neutral
   body there and its words take the hue (`--status-warn-word` / `-error-word`); status words retain their hue over
   the room's neutral or voice-local backing. Each theme has
   one ink polarity, dark text on every light surface and light text on every dark one,
   never both in one room. A state is a tone with its own ink, or a status mark and a word. Hue says identity and fill strength importance, so a
   mark that only identifies stays quieter than the name it marks. Crimson is the one
   brand accent: the chosen place, the primary action, a room's live word, and *new for
   you* on a status mark. Amber is the owner's attention (§4 Status marks). Colour lives
   on objects, never on the room.
4. **One control, three tones, one height.** Everything pressable with a body is a
   capsule standing on the rest shadow in every state (on the float shadow when it
   flies over moving content, like ↓ latest): *neutral* (the capsule surface)
   or *tonal* (its hue's fill and that fill's ink, Material 3's filled and tonal steps:
   the crimson for the primary action, the chosen tab or place and a count; the soft crimson
   for a chosen option or card, such as a filter, a segment, a switched-on capsule or the
   current step of setup; green for save, red for danger, amber for the owner's turn, which
   on a dark room are a neutral body with their hue's ink, principle 3). A
   *quiet* control (a link, a filter, an icon, a navigation row) has no body and never
   stands. A state changes colour, never height: hover deepens the fill, a press
   deepens it again, a chosen or switched-on capsule takes its tone and keeps standing.
   Inactive controls lose their shadow. A settled answer is a record: in dark its chosen option keeps the owner's
   sapphire lift and an explicit `Your choice` label; other options lie flat. No outlines. Shape follows content: a
   one-line control is a capsule; a choice that carries a paragraph (an offered answer)
   is a card with the object radius, standing like every other. A label that acts takes the
   primary ink; a setting's name or an unchosen option the secondary ink; a setting's
   value the meta ink, never the darkest; a chosen state its tone's ink, on the fill or, for a
   control that stays bodiless when chosen (Read aloud while it speaks, a mark), as a word on
   its own (`--tone-crimson-word`); a chosen choice in a menu (a model, a voice) keeps a faint
   crimson wash of the accent ladder under the crimson word, and a selected file keeps its own
   ink on that wash. A status chip lies flat: standing on a shadow means pressable.
5. **Air is spacing, never paleness.** Text keeps reading contrast; lightness comes
   from room between and inside objects and from the absence of lines. Nothing is
   faded to say "not yet known"; it says so in words or with an unlit mark.
6. **Colour answers touch.** A press answers with the capsule's own colour, never a
   sink or a lift; live work loops as three moving dots, the one status motion; a
   loader that is not work (history, reconnecting, a widget refreshing) keeps its own
   progress signal; nothing else animates by itself, and a state change never moves a
   neighbour, resizes a bar or changes a height.

**Seed (light | dark)**

| Role | Light | Dark |
| --- | --- | --- |
| Room · sidebar | `#ffffff` · `#fafbfc` | `#0f0f11` · `#0f0f11` |
| Object · tray | `#ffffff` · neutral 3 % | white 5.5 % · white 4.5 % |
| Capsule (hover) · float | `#ffffff` (`#f5f6f8`) · `#ffffff` | white 9 % (13.5 %) · `rgb(32 32 34 / .92)` |
| Glass on controls and sheets over content | none | `blur(18px)` |
| Owner · Ouroboros · task card | `#e9eef9` · `#fdf2f4` · `#fff8f9` | sapphire `#103e7f` · cherry `#76162b` · cherry `#76162b` |
| System, resting like every voice and without an edge | amber 12 %, name `#805000` | white 5.5 %, a gold name `#f7cb58` |
| Ink on every surface | dark (`#20232b`) | light (`#efeff2`) |
| Crimson word · ink on its fill · without a fill | `#bf2f40` · `#b52b3b` · `#b52b3b` | `#fe7d85` · white · `#fe7d85` |
| Crimson fill, strong · soft (hover) | crimson `#c93545` 10 % (14 %), both | `#d32d43` (deeper, `#c32a3f`) · `#872236` with ink `#ffdfe1` |
| The chosen place's left mark | `#c93545` | white, the fill's ink |
| A warning or failure on a row whose body cannot show it | the row's tint, its words in their own ink | a neutral row, its words in the hue's ink |
| Green, red, amber fills (hover) | the hue at 14/22, 10/16, 18/26 % | white 16/20 % (green), 9/13.5 % (red, amber), each with its hue's ink (red `#fe9498`, a step lighter than the crimson word) |
| Success word · its dot, track and switch | `#176b49` · `#22c55e` | leaf `#83e499` · `#2fd359` |
| Inside the Ouroboros voice: ink · secondary · meta · disabled | the room's inks (its prose `#35272c`) | `#fcf9f9` · `#ebd9da` · `#f2e4e5` · `#e1cbcc` |
| Inside the Ouroboros voice: capsule (hover) · tray, also the bodies of its tones, status chips and Project capsules | the room's | opaque `#8f263a` (`#9e3044`) · `#832336` |
| Inside the Ouroboros voice: crimson and failure word, link and focus · live word · Project ink · information ink | the room's | `#ffc2c6` · `#ffb3b7` · `#f6c4fc` · `#b6d4ff` |
| Inside the owner's voice: link, focus and quiet words | the room's | the owner's blue `#a0c6ff` |
| Chosen answer: fill · ink · detail · lift; its `Your choice` word | the green fill · its ink · meta · flat; in the accessible name only | sapphire `#235193` · white · `#dce9fc` · rest shadow; shown |
| Recorded answer: `You answered` word · its dot; the recommendation | meta · green; `#176b49` | the owner's blue `#a0c6ff` · `#a0c6ff`; `#ffc2c6` |
| Controls held while an answer is sent · a message awaiting its echo | 75 % · 78 % opacity | full ink; their words carry the state |
| Rest shadow | `0 1px 2px rgb(28 33 51 / .05), 0 6px 18px rgb(28 33 51 / .06)` | `0 1px 2px rgb(0 0 0 / .4), 0 6px 20px rgb(0 0 0 / .28)` |
| Float shadow | `0 2px 6px rgb(28 33 51 / .06), 0 18px 44px rgb(28 33 51 / .11)` | `0 2px 8px rgb(0 0 0 / .45), 0 18px 44px rgb(0 0 0 / .5)` |
| Press · on a solid fill under white | `brightness(.95)` · the same | `brightness(1.12)` · `brightness(.9)` |

- Radii by role, both themes: capsule `999px` (every one-line control and chip) · object `18px`
  (bubbles, cards, groups, menus, answer cards) · inner `12px` (trays and rows inside an object) ·
  sheet `24px` (the Shade's lower corners, the composer's field).
- Type: the system face (SF Pro on Apple), 12 / 14 / 16 / 24; weight 400 to read,
  500 for a status word, a root's title or a setting's name, 600 for labels,
  controls and a setting's value.
- Space: 4 / 8 / 12 / 16 / 24; 14 between messages; 12 x 18 inside a bubble.
- Motion: 120 ms for colour and shadow (`--motion-tone`); no press motion; the
  three-dot loop only while work runs. A surface that comes, goes or fades rides one
  curve, `--ease-sheet`, for `--motion-open` (340 ms, coming in), `--motion-close`
  (280 ms, going out) or `--motion-fade` (180 ms, a menu or the Shade; what leaves eases
  in); it moves by position, opacity and a small scale, never blur. Three folds change a
  height: the capsule row and the reply strip on that curve, and a deck of notes between works,
  whose height lands at once inside the reading write while its cards glide in by position and opacity. The handover between places
  keeps its own beat (`web/modules/shell_motion.js`); the expensive write of a change
  waits for its motion's end (`web/modules/motion.js`).
- Where it lives: the values are `web/ui.css` tokens, dark in `:root` and light in
  `:root[data-theme="light"]` (`--elev-*`, `--radius-object` / `-inner` / `-sheet`,
  `--surface-object` / `-object-opaque` / `-capsule` / `-float` / `-tray` / `-side`, `--glass`, `--voice-*`,
  `--tone-*` with `--tone-crimson-ink`, `--tone-crimson-soft` / `-soft-ink` and `--tone-crimson-word`,
  `--place-mark`, `--status-warn-word` / `-error-word`, `--press` / `--press-solid`); the light
  block restates its own values of the hues and tints the dark seed sets differently, and the dark voice materials
  follow it (one rule for the Ouroboros voice, one for the owner's) restating the roles they change; the recipes are ui.css "One
  control, three tones" and
  style.css "Airy dopamine", written once for both themes. A page maps its own
  elements onto these roles and adds no recipe.


The shell offers **Settings → Appearance → Light / Dark / System**. New clients
start on System; explicit Light or Dark remains pinned. The shared semantic
palettes in `web/ui.css` preserve geometry and status meanings. Both themes follow
the design bible above; neither has a decorative background.
On Windows the desktop window keeps its ordinary system caption, buttons, Snap and
resizing; only their light or dark tint follows the painted palette — the system
tint, not the page's colour — and a High Contrast theme keeps its own colours.
On macOS the main window is one surface: the page reaches the top edge under a
transparent title bar, the real traffic lights sit level with the header row, empty
parts of that band drag the window, and a double-click follows the Mac's setting:
Zoom, Minimize, Do Nothing or Fill. Fill relies on an available undocumented native
action, not an imitation by Zoom. Its appearance follows Light or Dark and inherits
under System. In full screen the bar hides until the pointer reveals it; the setup
and error windows and a UI that does not declare it keep the ordinary title bar. A
browser, another OS or an older desktop app reserves nothing.

Appearance belongs to a browser profile or desktop client, not an account or
server setting. Existing saved Light/Dark choices keep their meaning. Storage
failures are visible; clearing site data returns the choice to System. Switching
repaints mounted charts and diagrams without rebuilding views or losing drafts.
Independent iframe interiors remain author-owned, not automatically recoloured.
Desktop persistence requires a desktop app (launcher) built with persistent
WebView storage, 7.2.0 or later; in-app updates replace the core, never that app.
An older app opens its WebView in private mode, which erases website data each
time its window opens, so Settings → Appearance names that app by its own version
and says the choices reset at restart, instead of presenting them as kept. Server
restart alone cannot verify survival across full quit/relaunch. Mechanism and
deployment limits: ARCHITECTURE §3 “Navigation and shared UI contracts”.

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
- **The one reading exception: a full rich chat answer.** A chat bubble whose
  message is rich Markdown (an answer, the Project completion mirror, a
  `markdown: true` System row) is long text at the bubble's 16px reading
  size. Its headings step up from that text: `--md-heading-major` (1.25em,
  20px) for `#` and `##`, `--md-heading-minor` (1.125em, 18px) for `###`,
  the reading size for deeper levels; its tables read at that size too. The
  steps are relative to the reading text, not UI sizes, and only the rich
  answer's heading rules read the two tokens (§5) — they and a delivered
  document in the reader (§5 "Document reading"). Nothing else changes
  size: controls, labels, cards and page chrome keep the four sizes, and so
  do the compact surfaces (a Skill Review report, a task timeline) and the
  other rich-content cards (the question card, the update letter).

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
- A Markdown heading is a subsection heading too — a `--type-body` semibold
  label in a Skill Review report, the question card and the update letter,
  and its row's own size in a task timeline — except inside a full rich chat
  answer, which is reading text and follows the §1 ladder instead.

**The 12px UPPERCASE label pattern is retired** on migrated surfaces. All-caps
at a small size costs legibility, widens every label, and when a panel repeats
it dozens of times the labels collectively out-shout the values they describe.
Authored label strings should read as sentence case; CSS must not manufacture
caps with `text-transform`.

## 3. Foreground and state colour

| Token | Meaning |
| --- | --- |
| `--text-primary` | The one thing this row/card is about; the label of an action |
| `--text-meta` | Real secondary content: labels, notes, hints, meta lines |
| `--text-secondary` | A quieter step below meta: a dense repeated field; a setting's name or an unchosen option on a capsule |
| `--text-disabled` | Genuinely inert or incidental content only |

`--text-muted` is a legacy alias of `--text-disabled`; new work names
`--text-meta` or `--text-disabled` so the intent is readable in the diff.

`--text-secondary` is a real fourth step, not an alias: it sits between meta
and disabled and is written at ~50 call sites. Reach for it only when
`--text-meta` is genuinely too loud — a value repeated on every row of a dense
list, a chip's supporting word — and `--text-disabled` would be unreadable.
The criterion is the same one that governs the whole table: **if the owner has
to read it to act, it is `--text-meta` or brighter.** A load-bearing caveat, a
one-off note, a hint that explains a control: those stay at meta. A control's own
label is not content: it follows bible principle 4 (an action's label is primary, a
setting's name or an unchosen option secondary). A tone or an
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
that token, the named roles built on it (`--accent-light` for accent text, the crimson ink on light,
`--accent-dim` for a fill, `--focus-accent-border` / `--focus-accent-ring` for
focus), or a rung of the accent alpha ladder (`--accent-04` … `--accent-55`; in the light theme the crimson tone
and its hover are `--accent-10` and `--accent-14`, while the dark theme's crimson tones are opaque fills of their own,
listed in the seed table).
A new alpha is a rung added to the ladder, never an `rgba()` literal in a rule:
the ladder is what makes "make the accent calmer" a one-line change instead of
a grep in the light theme. The dark theme's crimson roles (the tones, the crimson word, the
focus border and ring) are values of its own seed rather than rungs of the ladder, so a new
accent rewrites those few tokens there too.

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

**Secret visibility has one meaning across Settings.** Show reveals the complete
current value as selectable, read-only text in the masked field's own place,
never as a second field; long values wrap within the column. An edited value is
shown from the local draft; an untouched saved value is read only when requested.
A prefix or a configured-value placeholder is never a successful reveal. Hide
brings back the unchanged masked field and its draft; typing a new value starts
with Hide or Clear. Viewing changes neither the dirty state nor the values
submitted by Save or Test. Clearing, renaming the entry, reloading or leaving the
page closes the reveal, and a late read cannot reopen it. Built-in, custom,
skill-requested and MCP header fields use the same interaction; loading Settings
keeps saved secrets masked. Pending reads and failures are shown at the field.

A segmented choice row gives every choice the same width. Up to four fill their
row; a longer scale wraps into equal columns of four and leaves its last row partly
empty rather than stretching a lone choice; a narrow card stacks two; one-glyph
values such as review cycles stay one row. An empty inherit choice is as real as
any other, and the focused choice wears the one focus ring. A choice the running
Access suspends — Blocking under Cyber Pro — stays selected; the note beside it
names its effect for the current process and for the selected next start, from
the draft before Save.

Short fixed choices keep native selects, including the platform's own popup.
A control never widens its column: a select shows its chosen label on one
line, clipped at its own edge, and the full label stays in the platform's
popup.
Model selection uses the shared editable chooser: suggestions assist typing
without becoming an allowlist. A saved unknown model remains editable; a
catalog refresh preserves the real input, selection and composition. Escape
or blur closes suggestions without assigning a value. Selected, hover, focus,
disabled and invalid states have different meanings and remain distinguishable.
Availability never rewrites a selection: a Claude `[1m]` variant whose base
model a usable account lists reads Available, noting that the engine checks the
variant at session start, and an unread catalog reads Not checked. Maximum
response sits beside the context window as Auto or a whole number for this
exact model, server and account; Apply acknowledges it outside the Settings
draft, and a route change clears it. Web search is its own Source and Model:
Auto lists the routes it may use, a chosen source is strict, and skills, MCP
and browser tools stay independent.

Tabs expose one selected view and one keyboard entry point. Arrow keys and
Home/End move through available tabs; restoring a selected tab reveals it by
scrolling its strip, without moving the page or taking focus. Menus and
editable suggestion lists share viewport placement, not keyboard semantics:
a menu moves focus among actions, a chooser keeps it in the input. Dialog
focus stays in the modal context and returns on close when the caller remains
available. Popups sit outside decorative clipping and fit the usable viewport.

A **message field** — the Main and Project composer, a question's own answer —
sends with Enter by pressing its own Send action, whose validation, disabled
and pending state decide what is sent; a message is sent once. Shift+Enter
breaks the line; Ctrl, Cmd or Alt with Enter send like Enter. Enter sends
nothing when it commits an input-method composition or the field is empty,
disabled, read-only or settled, and once when held; a rejected answer keeps
its draft. A message field grows with its text or placeholder to one cap, then
scrolls. Touch keyboards are asked for a Send label; support varies. Every
other multiline field — file editors, JSON, settings and widget forms — keeps
Enter as a line break.

### Composer owner controls

Each composer saves its room's **Context** and **Effort** through the owner endpoints.
Main supplies a one-time copy to new Projects; later changes stay in their room.
Readback comes from Main state or the named Project, never another room. Saves apply
to new participants; running work retains its choices through Resume. A refusal
keeps the saved value and says why. No scope selector or extra settings row.

Each is a neutral **capsule** of the pills' height — ring, name, value: `Context Max`,
`Effort X-High`; the name in secondary ink at 500, the value in meta ink at 600,
and only the ring's lit arc coloured (crimson; Low neutral ink, Nano the Project's): a level is never amber.
Until the room's values arrive the ring stays unlit and the value secondary,
never a faded capsule. Closed, Effort names only the primary level; its ring (a 300°
arc: band minimum..maximum, dot primary) and the button's name carry the range.
Context's ring is three arcs, the chosen one lit: one of three modes, never a fill
level or capacity. A capsule opens **inline**, never as a dropdown, one at a time;
its value yields to the strip. Context's is Nano · Low · Max under a glass thumb; a
choice posts at once and the strip waits for the answer. Effort's seven levels carry
a primary pill and two helper bounds; the primary may lie outside them. A
segment is its word plus `--capsule-seg-pad` a side; the pill covers its segment;
band, pill and brackets share one vertical inset. The primary and helper bounds never
move each other; the brackets cannot cross. A level tap sets primary; each bracket
has its own handle. A drag keeps its grab offset on whole levels. A stored `minimal` shows at Low until its handle moves.

**A mark keeps from a rounded end the clearance it keeps from the straight edges.**
A pill (the thumb, the recommended pill) is concentric with the capsule, so the band
inset suffices; a square bracket at the last level would graze the curve, so
Effort's end gutter also clears it. The gutter is measured into the fit and outlasts
scrolling and dragging.

**The open strip stays on the composer's line**: beside the others when it fits
(padding down to `--space-1`), else alone while they step aside (down to 3 px, the
pill's corners tightening; the name and divider yield first), last scrolling inside
itself, faded where levels hide. A closed line too narrow drops the names, then
ellipsizes the longest values, never rings. All is measured, so no breakpoint.

**It opens in place.** Beside the others the strip grows by its width. Alone, its
glass grows from the closed capsule to the whole line in 300 ms on the iOS sheet
curve, the ring riding on the glass's leading end, while the controls that step
aside fade where they stood (110 ms) and the levels appear one after another;
closing runs back in 220 ms. Only opacity, a clip and the ring's offset move, so no
frame wraps (`web/modules/capsule_morph.js`); reduced motion shows the strip at once.
Inside Telegram a level the finger reaches, and another level or mode pressed, ticks
(§10, Haptics).

**On a phone the row folds while the owner reads back.** Scrolling back through a
room's history folds the row of capsules away: its line collapses, the capsules
fade, and ☰, still the row's first control, waits beside the field on its last line, beside "+". Scrolling
toward the latest, reaching it, focusing the field or another layout brings the row
back; while the owner writes or works a capsule, or the field holds the focus, it
stays. Only the owner's own scroll decides, as with Safari's bars: older history
paging in or let go, and rows that arrive or grow, move nothing; nor does a swipe
begun in a card's frame, which the feed never sees, or the app's own jump (the
waiting pill to a question, which keeps the pill under the finger for the next one)
(`web/modules/composer_fold.js`; owner 19.10).

**Hover is an accelerator; the press always works.** With a hovering fine pointer a
capsule opens ~160 ms after arrival (not while the other is open) and closes a
second after it leaves a 32 px zone unless a press pinned it; the press unpins.
Touch and keyboard use the press: Enter or Space opens onto the recommended handle
or the chosen mode; arrows step a handle (PageUp/PageDown too) or choose the next
mode, Home/End reach the ends; Esc closes and returns focus to the head; an outside
press closes. Handles are sliders naming their level; closed, they leave the tab
order. An effort change saves when the gesture ends, saves run in order (last wins),
and Send waits for both pending Context and Effort saves; a refused
save keeps the draft, says why and returns to the saved value — unsaved values are
never shown as saved.

**"+" holds the composer's actions.** Where the paperclip stood, "+" (More actions, a
finger-sized target that keeps the row's height) opens a menu sheet above itself: Attach
file first, then the sections later controls add, each under its label; drag and paste
still attach, and a pointer that opens it from the field leaves the focus in the field.
The dictation mic has a slot between the field and Send that takes no room until an
engine is chosen. Contributed chips have a slot at the end of the capsule row that takes
only the room the capsules leave (never a second line: the chips that do not fit fold
into one "+N" whose list holds them) and steps aside with them.

### The composer field

**One object whose corner never grows.** The field is one floating object: a staged
reply (Replies) is its first line, and the text row with "+", the words, the mic and
Send its last. Its corner is the sheet radius, half its one-line height, and stays as
the words grow: one line is a capsule, a paragraph a card with the same corner, never
an oval. "+", the mic and Send ride on the words' last line, never their middle. The
reply panel inside is concentric (the sheet radius less the field's 8 px inset) with no
hairline of its own; the text row has no surface; the field takes the float shadow and
the focus ring, drawn as shadows, so a border never adds to its height (owner 20.10, 1C).

**A Send rises into the feed.** An accepted Send lifts a copy of the new bubble from
where the draft stood, its words on the draft's, to the bubble's place in 340 ms on the
iOS sheet curve: its box moves and resizes, never a scale, while the field's surface,
corner and shadow turn into the bubble's, and its sender line grows in; the bubble
shows as the copy lands. A bubble that moves under the flight (a phone's keyboard going
down after a tap on Send) is followed to its place; a room that goes away (the drawer
over it, a closed Project) ends the flight at once. Reduced motion, a hidden page or a
bubble outside the feed's view: it simply appears (`web/modules/send_flight.js`; owner
20.10, 2A).

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
Primary (16.7:1), meta (9.6:1) and secondary (6.8:1) clear the 4.5:1 floor against
`--bg-primary` with room to spare. `--text-disabled` only just clears it (4.7:1) and is
therefore reserved for genuinely disabled or incidental content; it must never carry
meaning a reader has to obtain.

### A selected state is not exempt from contrast

A **status hue** and a **status foreground** are different values, and the
selected state of a control must use the foreground. Selected Advisory in the
enforcement group once read `--amber` (`#f59e0b`) over a 12% amber wash: ~2:1 on the
light surface, unreadable exactly when the owner had chosen it. The rule that
closes this: a selected control takes a tone, a fill with its own ink that each theme
defines (a chosen option the soft crimson, `--tone-crimson-soft` with its `-soft-ink`); a
status colour reaches a control only through the `--status-*-fg` / `-bg` / `-border`
triple, also defined per theme, and never through the raw hue token, which is not. A
setting value carries no status tone at all: Advisory now takes the shared chosen-option
tone (Status marks).

The same reasoning covers images. A colour baked into a `data:` URI cannot be
themed, because a custom property cannot be interpolated into the URI string —
which is why the select chevron was a pale `#e2e8f0` on white. **The whole
image is the token** (`--select-arrow`), overridden per theme, not the colour
inside it. `tests/test_appearance_static.py` holds both facts.

## 4. Status and chips

A status has **an explicit foreground/background pair**, never a foreground
derived from whatever generic opacity sits on the element.

The connected Chat header says `Starting…` until a complete server-ready
observation confirms the supervisor can assign work; a live socket alone means
only that the UI transport is open. The host's boot phases refine the word
without changing the fact: `Chat ready, restoring background tasks` once the
dialogue accepts messages durably while pooled work is still not admitted, and
`Start failed` (its cause in the badge title) when the generation cannot become
ready. `Online` is reserved for a ready, idle
runtime; known queued, active or finalizing work keeps its own factual label. A
durably accepted incoming message likewise proves acceptance, not that its task
has begun or a model has read it.

The Update indicator names the real stage, then the last confirmed outcome or explicit uncertainty. It opens the existing Updates panel, never a second apply flow. The owner may dismiss that result from the indicator only; the panel retains it until the next admitted attempt, with actual saved-local-work references and accessible diagnostics. A pre-admission refusal stays with the request. A busy-work refusal offers explicit retry, never a delayed automatic restart. Connection or version alone proves no successful update. Unknown remains neutral; confirmed completion uses the success triple and a failed/refused/rolled-back attempt the error triple.

Status, owner action, and urgent notification are separate product concepts:

- **Status** states a fact about the affected object, never implying that the
  owner can or must act. Task status uses one factual word family, `Working`,
  `Done`, `Done with warnings`, `Failed`, `Stopped`, which is also the host's
  durable label vocabulary for its own task rows in Main and the Project thread
  (`OUTCOME_PHASE_HEADLINE`), not only the browser's.
- **Owner action** exists only when the responsible domain exposes a current
  concrete continuation, such as Resume, Retry, Connect, Repair, Grant access,
  or Restart: a real adjacent control, which severity alone never manufactures.
- **Urgent notification** is a rare, time-sensitive interruption. It uses the
  product's explicit incident/notification seam, not a red status or a failed
  task as a proxy.

Activity schedule rows use the same factual status rule: `active`, `disabled`,
`consumed once · history`, `withdrawn` and `deletion pending · work in flight`
describe lifecycle state, while the adjacent Disable/Enable, Restore and Delete
controls state the owner action. Lifecycle and the last run's outcome are
separate facts: the outcome line says the settled word as stored (`completed`,
`failed`, `cancelled`), `ran · no message` for a quiet run, or `outcome not
observed yet` until the scheduler has read the settled result; it never says
running, and only the host's own observation writes `previous run still open`.
A consumed one-shot is history whether its task succeeded or failed; the status
never implies a result. A row is headed by its name; the instruction stays
readable and expands by touch, and a waiting row says why inline, never in a
hover-only tooltip. Retained rows (consumed and withdrawn) collapse into one
History & withdrawn disclosure rather than padding the standing list or
disappearing: history the owner can still open, read and act on. A withdrawn
row says who withdrew it, when and why, and keeps Restore through the retention
window once no hold remains on it; Restore cancels the withdrawal and brings
the row back enabled, even if it was paused before, but never re-arms a one-shot
that already fired. A consumed row keeps only Delete, since Enable on a schedule
that cannot fire again would be a lie. A withdrawn row whose accepted work is
known to be in flight stays in the standing list and says so; an exact hold release (Restore
hold) stays separate and preserves the withdrawal.

A record of what the owner did earlier is history, not status. A skill's
OuroborosHub submission (the version it sent and its PR link) lives in the
card's details disclosure without a tone; it never claims the PR merged or gates
or replaces the card's current action, and forgetting it is an explicit local
choice, not the way to unlock one.

A task-bound `Reviews` history row may be its owner's only retained fact: it
keeps a neutral owner anchor visible but hides task status and typing until a
real task status or activity arrives; review presence alone never means
`Working`, `Done`, or owner attention.

A review only awaited when its task ended is not a warning: reviewers that had
simply not answered yet leave the task `Done`. For a plan review the task result
keeps the host's typed disclosure that the review was still open (the
`terminal_host_notice` field, read by the CLI and by parents), the card states
the same fact as its cause sentence, and no second chat bubble carries it; a
card that reads `Done with warnings` or `Failed` for another reason keeps that
reason, with the open review stated beside it when the result records it. For a result review the card's
cause sentence already says whether the reviewers signed the answer off or
Ouroboros delivered it on its own judgement. A real
outcome keeps its word: no reviewer quorum, a failed, refused or unresolved
slot, a collected blocking finding, a reviewer verdict nobody closed, a rail or
a blocking exit still read `Done with warnings` or `Failed`.

A reviewer panel that settles after its task already ended adds one System row
naming the reviewed version (delivered, different, or delivery unknown) before
its verdict. Its reviewer lines name the model that answered as an engine
reported it (else the requested one, labelled so: a direct API route's own
target is a request, not a report; two seats of one model stay two lines) and
say each outcome in words: passed it, rejected it, inconclusive, still awaited,
outcome unknown, not sent, unavailable, or answered without a readable verdict;
an unavailable reviewer's cause is quoted as the delegated engine's report,
never as the provider's. A reviewer's shortened note, like an engine cause the
host's own bound shortened, says only that it was shortened; the row offers the
exact applied record as a download once the host stored it, and a cut of the
whole row names that record only then, never a guessed link. Slot ids, verdict
tokens and omission markers stay in that record and the model's mailbox. The row
lands inside the finished card (its Reviews group carries the note, the timeline
keeps the row) without changing the card's chip, title or meta; a standalone row
appears only when the task has no card record in the page. Saving and showing
late criticism does not replace passing it to Ouroboros: the host owns sources
and fact delivery; Ouroboros judges the evidence, explains it and chooses any
further work on an admissible turn. The untyped terminal host notice and the
origin-addressed routing notices stay ordinary rows by design. Local diagnostic
failures stay inspectable in details and Logs but do not relabel the whole
still-working task. A failed child keeps a compact factual `Failed` marker
inside its parent while the root continues under its own authoritative status.
Internal reason codes belong in details and diagnostics, not compact headlines:
a card that shows a cause says it in the owner's words while the record keeps
the machine code. In root ending summaries, episodes and Main completion rows, a typed cause
without a sentence reads "The recorded reason has no description yet", with
its exact code in accessible details. Authored warning prose and raw host text
stay intact; never infer meaning from spelling. The loop's forced-finalization rails
(round limit, deadline, grace window, context, unabsorbed children) and the
supervisor's timeout reaper (maximum running time, deadline, idle silence)
keep typed codes on the record and incident key, and the
card, the reaper's grace toast, kill notice and salvage line, and the loop's own
fallback text all say the one sentence from the shared table
(`project_dialogue.TASK_CAUSE_PHRASES`, whose browser twin lives in
`task_reason_words.js`). The routing receipt under an owner message is such a surface:
a refused addressing act carries the host-composed `cause` sentence
(`project_dialogue.routing_refusal_cause`, one host table for the receipt line,
the System row and the picker toast), a landed act carries none, and an unknown
reason stays raw. A provider failure quotes only the provider's own sentence (a
long one, or one its producer already cut, shortened in words beside the Logs
pointer); without one it names the host's classification of the failure (beside
the provider's own error code and parameter when an engine relays them, as the
provider's) and points at the Logs, which keep the host's record of it (a stream
error's sentence only as a bounded excerpt, so no row promises more): neither an
exception's text nor an engine's composed diagnostic reaches the owner's row.
Host text speaks only for the host's own actions, counts and signed quotes; a
source it could not read is unknown, never zero. A terminal whose preserved
output was never reviewed shows that output labelled, not hidden: a short
labelled excerpt beside the pointer to the full copy, so a `Failed` card over
applied work is never a bare headline and never names preserved bytes without a
way to reach them. Where a stop receipt already carries the same text in the
very chat the card is written to, the card keeps the label and pointer alone; a
card in another chat keeps the excerpt.

| Role | Foreground | Background | Border |
| --- | --- | --- | --- |
| Success / connected | `--status-ok-fg` | `--status-ok-bg` | `--status-ok-border` |
| The owner's attention: the owner's move, or a warning worth reading | `--status-warn-fg` | `--status-warn-bg` | `--status-warn-border` |
| Error / failed | `--status-error-fg` | `--status-error-bg` | `--status-error-border` |
| Neutral / classification | `--status-neutral-fg` | `--status-neutral-bg` | `--status-neutral-border` |

- **Status renders as a mark + text** (Status marks below). The mark carries the
  state at a glance, so the sentence need not shout it in saturated colour and
  sits at ordinary reading contrast.
- **A known outcome owns the status; an unfinished lifecycle stands beside it.**
  When a task's outcome is settled while post-task work still runs, the chip
  states that outcome from the five-word family and `Finalizing…` is a SECOND,
  quieter fact next to it: not a replacement, not a sixth status word, never
  something the card title carries instead. The pair is one accessible name, so
  a screen reader hears the outcome and the hold together. An owner stop
  outranks the hold entirely and speaks for itself. Dashboard → Activity applies
  this to a row's own outcome, never its root's: plain `running` yields to it, a
  queue wait or hold (`pending`, paused, sleeping, held after Restart) stays
  beside it as the second fact, and the row keeps its controls while the task
  still occupies the queue.
- **Neutral is a real state**, not an absence of one. A classification chip
  (which agent, which family) is neutral: a tag, not an alarm. A tone value the
  code emits (`muted`) must have a rule; falling through to a default is how
  chips end up white.
- Chips are `--type-meta`, not smaller, and are not uppercased.
- `--green` / `--amber` / `--red` are the saturated hues, for things that are
  not text: dots, switch tracks, progress. The `--status-*-fg` tints are for
  text on near-black; do not swap them. (A `--tone-ok` / `--tone-warn` /
  `--tone-danger` alias family, plus `--accent-task` / `--accent-system` /
  `--accent-user` / `--accent-project` and `--ui-tone-*`, were named here and
  referenced by nothing, so every surface invented its own literal instead. They
  are gone; the vocabulary above is the whole vocabulary.)

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
  "connected" the loudest thing on the panel. A room's status beside its title is
  its word and status mark (the sentence's own green dot only at rest), with a caret
  when it opens, never a filled chip.
- **A status chip** — `.ui-chip[data-tone]`. The full triple, because the chip
  *is* the status and has nothing else to carry it.

Surfaces whose tone is a class suffix rather than `data-tone`
(`.toast-*`, `.log-phase`, `.widget-table-status`,
`.widget-metric` / `.widget-callout`) name the same tokens. A task card's chip
(`.chat-live-phase`) and a settled Logs task headline keep their phase classes as
facts and take their tone from the `data-chip-tone` attribute their painters write. A surface that
paints only its edge (a card tinted by its state, a callout's left rule) takes
the border and leaves its own background alone; a toast keeps its glass
background, because a translucent status fill over live page content costs the
text its contrast.

Adopting these tokens applies the semantic status contract, which already
governs every surface; it is not a token migration of those surfaces and does
not move them into the migrated set in section 8.

### Status marks

Every surface that says what a work or a room is doing draws one small family of
marks (`web/modules/status_mark.js`, the `.status-mark` recipe in `web/ui.css`): a
Project row, Main's row, the Projects header, ☰, a room's status, a Shade row, a
task card's chip, a question's head, a skill's job. A mark carries two facts in two
channels, and nothing else draws a work's or a room's status dot, bar or ring (a
status sentence's or a composer chip's dot is its tone, §4).

- **The shape says what the work does.** For projected work the server names it
  (`mark`, `ouroboros/gateway/work_status.py` `mark_of`); the client adds only the
  four facts it alone knows (Room work status).
  - *moving* — three 4 px dots 3 px apart in the 1.15 s typing wave (each lifts 4 px and a little larger,
    150 ms after the one before): work runs;
  - *still* — the same dots at 40 %: queued, or a wait that ends by itself;
  - *pause* — two 2 × 8 px bars at 70 %: paused, by the owner or by Ouroboros, until Resume;
  - *ring* — a faint 7 px ring: not confirmed (an incomplete read, a closed socket), never a guess;
  - *dot* — a 7 px dot: one fact (the owner's move, new messages where nothing works, an outcome);
  - *none* — rest draws nothing.
  The ring stands at 55 %; new messages lift the still dots and the ring to 60 % and the bars to
  full strength, so the crimson reads.
- **The hue says whether there is something for the owner.** *Ink* is the
  surface's own ink (`currentColor`), a chosen row's selection ink included, so a
  mark follows the name it marks and never out-shouts it. *Amber* (`--amber`) is
  the owner's move: a question to answer or an action only the owner can take; it
  outranks everything and stands alone, the dot replacing the work's shape while
  the word keeps saying the work (`Working` with a small amber dot). *Crimson*
  (`--accent`, its light crimson in dark) is new messages: it tints the work's
  shape, or is the dot where nothing works. An *outcome* is a dot beside its word:
  green for Done and Done with warnings (the word says the warnings), neutral for
  Stopped (someone's decision), red for Failed.
- **One mark per place.** A row, a bar, a chip or a head carries one mark; the
  owner's move comes first, then an outcome beside its word, then new messages,
  then the work's shape in ink. A lone dot in a list is only new messages or the
  owner's move.
- **Words carry the meaning.** A mark is decorative (`aria-hidden`); the word
  beside it or the place's accessible name says the same fact ("Working · new
  messages", "Something here waits for you"), so colour is never the only channel.
- **Motion.** Only *moving* loops; reduced motion stops the wave, and the moving
  dots then stand at full strength, apart from the still ones. A mark that changes
  shape on a live chip keeps the moving dots' width, so the word beside it never
  moves. A status sentence's own dot (`.ui-status`, Settings sentences) is drawn by
  its CSS at the same 7 px in the same hues; a room at rest says `Online` beside the
  shared mark's green dot. Both speak only at rest.
- **Amber is the owner's attention and nothing else:** (1) the owner's move — a
  question, or what only the owner can do: sign in, a spending limit, Resume after a
  restart, crash or Panic, grant access, a skill review that must run first, local
  changes before an update, an Evolution chip or Consciousness sentence that needs the owner
  or was stopped by the spending limit; (2) a warning worth reading — warn toasts and sentences,
  Settings and onboarding notes, warnings inside task timelines, reviews and Logs
  events, account quota; (3) the System voice (a pale tint in the light theme, its name
  in the dark one), a voice, not a status; (4) a few data categories that are not statuses (Logs and Storage
  categories). What goes on by itself or what someone chose — queued, waiting,
  paused, pausing, stopped, unconfirmed, work in flight, a mode, a level, a setting
  value — is never amber.

In a Project row the mark owns one reserved trailing column, so it sits at one x
across rows whatever the name length and, with a fine pointer, while the ⋯ is
hovered, focused or open (on touch a focused ⋯ brings back its own column);
Main's row puts its mark at the same x, its count just before it. A deleting row draws only `Deleting…`, its
census fact kept in the accessible name. The Projects header carries one mark in the
same column in both states, its count just before it as Main's is: amber when a
Project waits for the owner, else the strongest work among the Projects in ink. Its
chevron follows its word, and + takes the ⋯ column; on touch, where rows have no ⋯
column, + follows the chevron instead, so numbers, marks and actions each keep one
column (owner 19.10, Q1 = B). An activity repaint keeps the row and menu
nodes. A mark never carries a counter, percent or text animation.

Enforcement: `web/tests/status_marks.test.js` pins the mapping and that only
`status_mark.js` builds the anatomy and writes its shape and hue;
`tests/test_status_marks_static.py` lists every stylesheet rule that paints amber
under one of the four categories above and fails on a new one;
`tests/test_status_marks_*_browser.py` measure the marks in Chromium and WebKit.

### Unread

New messages are crimson, and only where the owner reads. The two sidebar numbers
count the same thing, the standalone messages not read yet (below): Main's on its
row, shown while another place is, and the Projects header's, summed over its rows,
each a crimson pill in one column just before the marks (owner 19.10, Q1 = B). A
Project row's mark takes the crimson tint (its work's shape, or a dot where nothing
works), and ☰ carries a crimson dot while Main or a Project hidden in the drawer has
some. Project names keep one ink, read or not. Main's count is kept as a Project's
is, its room's revision against a shared read cursor read the same way (below), so
every window and device shows the same number; a cursor no client has set yet starts
at what Main holds.

For Main and a Project, the crimson number and mark mean *there is a new message in
this conversation that you have not read*, never *something moved*; work marks,
notifications, toasts, task controls and Stop/Panic keep their own meaning,
unaffected by it.

Unread revisions follow successfully stored conversation rows: a failed write
keeps the existing live-delivery behavior but cannot advance saved history's
revision, so this counter guarantees no delivery durability.

- **Counts:** each new standalone message in the conversation — an ordinary or
  proactive reply, a question, a root task's final answer, a standalone System
  message (a root task's terminal incident included), a delivered photo, video,
  file or link card. A question or a delivery is shown alone and counts whichever
  task sends it, a child task included.
- **Does not count:** anything that changes a task card rather than the
  conversation — narration and progress, lifecycle, incidents raised in a
  task's progress (root worker failures included; they still toast), host rows placed in a card (custody, late
  review results, merge receipts, a child task's terminal incident), and a child
  task's own words and final: a child speaks to its parent, and its card shows
  them. Where a row is shown decides, never its kind: an incident shown alone as
  a System message counts, one shown in a card does not. A placed row is card
  content even where the page has not loaded its card and shows it alone.
- **Read:** only after a covering history read paints the newest-arrived message
  on screen in the visible room, clear of header and composer; being at the
  bottom is not enough when later cards or owner messages buried it. Opening an
  unread room therefore lands on that standalone message wherever it lies: until
  a loaded page holds it, the opening reads older pages (the page limit bounds
  what stays loaded, not this reach); a read room opens at the bottom. One
  opening intent, ended by a gesture (in a bounded box too), a question, leaving
  room or window or a failed page (bottom and Retry), never a jump on later
  reads or new arrivals. A question opened from its Main mirror has priority;
  showing it reads no newer messages elsewhere. Collapsed cards need not open.
  The newest message is the one that arrived last: a late answer keeps its
  written time and chronological place. While reading elsewhere the crimson mark remains
  until that message is shown by scrolling, `Scroll to latest message`, or a
  return to the page/window; opening alone never clears it. Unknown newest
  arrivals (an unfinished line, a damaged row) acknowledge nothing and are never
  searched for; state refreshes recheck an open unread room until history is
  repaired or newer standalone content arrives. A room with no standalone
  message is read at the bottom only once its clean start is reached (by bounded
  search, the opening or `Load more history`). An old child's final whose
  lineage appears only on another page can keep the crimson mark once the room is shown.
- **Several clients** share one read cursor that only moves forward: a room read
  on one client stops showing the crimson mark on the others at their next state
  refresh, and nothing read is shown unread again.

### References and actions

One owner intent has one control, built in one module. Tokens and primitives
cannot guarantee that alone: every copy of a control can reuse a primitive while
its caller still chooses the words, glyph and class, which is how "take me to
this Project" came to be drawn six ways. The rule is therefore about intents.

- A **reference** points at an entity that exists in the product and goes there
  when pressed. A Project is referred to by the Project reference (the Projects
  glyph, the Project's name, `↗`) and by nothing else, wherever content points
  at it. `project_reference.js` builds it and alone raises `ouro:open-project`;
  a caller chooses a layout (`inline`, `bar`, `footer`), never a label or a
  class. It names the Project wherever its row carries the name, even beside
  same-named work; a Project never named, and the owner's routed message (whose
  caption names the destination), read `Project`, never an id. It claims no
  activity (the work card's chip does). Its accessible name says in words what
  the glyph and the arrow say in pixels (`Open project <name>`); nothing depends
  on hover.
- **Two forms, one recipe.** The reference is the full-width strip that closes a
  work card bound to a Project (the body stays the work's), and a capsule
  everywhere else: under a mirrored answer or a System lifecycle row (in the
  shared `createSystemMessageActions` row, token spacing, wrapping, focus-ring
  clearance), at the end of the owner's routed message's one receipt line
  ("Passed on to «work» in", "Started as a task in"), and in a Main question
  copy's voice line beside `Ouroboros`, at that line's 28px height and meta size
  (`.ui-hit-area` keeps a finger's target; under a finger the line leaves the
  head's × its column, so neither takes the other's). The whole capsule is the
  target, with one fill whose hover and press deepen the whole capsule; no
  border, ring or hairline, and no width that depends on where it sits. Inside the dark
  Ouroboros voice its fill is that voice's capsule step and its ink a lighter Project ink, so the
  name stays readable at rest, hover and press. Long
  names: two lines in the strip, ellipsis in the capsule, never smaller type;
  accessible names stay whole.
- A **command** is a button: it changes or confirms something, or goes to a
  place that is not such a reference (`Load older messages`, `Open widgets`).
  Tabs, toggles, menus and the navigation list keep their own roles; a status
  chip states a fact and is not a destination.
- An intent that will be drawn in a second place gets its door first (siblings:
  `docs/inventories/UI_CONTROL_TEXT_INVENTORY.md`).

### Chat authorship and System rows

Voice follows authorship, not severity or transport. Model-authored replies use
Ouroboros's assistant voice; host-composed commands, receipts and diagnostics use
System voice. A **System row** is a chat message with `role="system"`, in the
yellow/amber system treatment or its task-card placement. `role` alone selects
voice; `system_type` names the kind, not the author (the Project question and
completion mirrors are System pointer rows showing model-authored bytes in
Ouroboros's voice). New host producers stamp both fields; relays keep them through
delivery, persistence and history. Model narration, proactive replies and
questions stay model-authored even when typed.

A host fact has a subject, and the subject names its carrier: a task's fact is a
row of its card, a fact about the owner's message is the typed annotation on that
message, and a landed act publishes no row (its receipt or card already says it).
A standalone System row exists only where nothing else can carry the fact: no mind
can speak, no mind will read the message, the owner's own direct act, the
runtime's state, Main's Project lifecycle, or words signed by their author (a
reminder, a skill notice). It is one short typed mark in owner words, never a
paragraph or text written for a model. `tests/test_host_message_voice.py` keeps
every host type stamped as a literal at its send site in one of these classes, with
its counted producer sites. Room status carries connection loss; reconnecting adds
no feed row or counter in Main or a Project (`tests/test_reconnect_feed_browser.py`).
Only the first open after a reload onto changed code shows `Restart complete`.

Language follows the same line. The install's interface language
(`OUROBOROS_UI_LANGUAGE`, Settings → Appearance → Language) translates chrome
and System voice at render from a translation memory the install generated or
imported, never from a dictionary in the repository: status words, cause
sentences and question status by their stable code (`tr`), labels and help copy
by their rendered English (the overlay), a relayed host sentence by its exact
text (`tx`). Model-authored text, owner-supplied names, logs and code are never
translated, and durable rows keep the English with their typed codes; a string
the memory lacks stays English, never blank or guessed.

Authorship also has a time axis. Ouroboros's voice means *the model, now*. Words
the mind wrote earlier for a later moment (a note left with
`schedule_followup(notify=true)`) reach the owner at that moment as a System row
(`system_type="reminder"`), never as a live reply: the host prepends one
signature line, `Reminder · Ouroboros · written <when> · for <when>` (plus
`· delivered <when>` when downtime held it back), and keeps the words verbatim,
never rewritten, summarized or re-timed; the next turn reads the row as a host
fact (`📋 [reminder]`), so Ouroboros knows what the owner was shown. A skill
holding the owner's `notify_owner` grant speaks the same way in its own name: a
`skill_notice` System row signed `Notice · <skill>`.

Formatting is asymmetric: assistant text always uses the sanitized chat markdown
renderer regardless of `markdown`; ordinary System text is escaped unless
`markdown: true`. The typed `skill_review` row keeps its dedicated renderer.
Voice does not confer task finality. The existing untyped terminal-host-notice
contract remains a documented exception, not a pattern for new notices.

Built-in chrome uses words and monochrome `currentColor` vectors, not decorative
emoji. System keeps its resting body and amber/gold label without an edge; status stays explicit in
words and the tone primitive. Functional icons keep state and accessible names.
Known typed host prefixes are hidden only at display (`host_notice_text.js`);
unlisted or ambiguous records remain literal. History, keys, exports and the
existing Copy input are unchanged (Copy already uses translated text for two
notice types). Authored text, quoted evidence, names, documents, logs and author
widget icons/interiors are not chrome and keep their characters.

Main's empty-chat greeting speaks in neither voice: it is host copy in a quiet
`.chat-empty-welcome` placeholder (a `Welcome` label over one sentence), never a
bubble, a history row or a model reply (`web/modules/welcome_preference.js`). It
appears only after a successful recent history read whose own window reports
complete coverage, over a feed with no message or task card (the typing indicator
and the ephemeral `Restart complete` row are chrome, by the same rule that gives an
empty feed its history loading state). Every later read withdraws it while in flight, so
that loading state, then any failure, shows instead; only a complete answer brings
it back, and it leaves with the first message or task card. Project rooms never
show it.

The sentence is a hidden install-wide preference with no Settings control: the
`welcome` key of `state/ui_preferences.json` under the data root
(`~/Ouroboros/data` by default). Change only that key and keep the file's others:

```json
"welcome": {"mode": "default", "text": ""}
"welcome": {"mode": "hidden", "text": ""}
"welcome": {"mode": "custom", "text": "Good morning."}
```

`default` shows the built-in "Ouroboros has awakened", `hidden` shows nothing, and
`custom` shows `text` as plain text, never markup (nonblank, at most 500
characters; other modes retain but do not display valid text). The object has
exactly these two keys. Prefer `POST /api/ui/preferences` with `{"welcome": {...}}`:
it validates and merges the value (400 on refusal, without writing). For a hand
edit, stop Ouroboros, back up the file, and preserve valid whole-document JSON
and neighboring keys. An invalid welcome inside valid JSON falls back to
`default` independently; the next save stores that default. Malformed JSON
instead follows the existing whole-file fallback: all preferences read as
defaults, and a later save may replace the unreadable contents. Main reads the
preference when it opens and on every reconnect; there is no file watcher.

## 5. Card and section composition

- A panel is one `.ui-card`-family surface. A card on the room is an object
  (`--surface-object` on `--elev-rest`, `--radius-object`, no border) and a group
  or row inside it a tray (`--surface-tray`, `--radius-inner`); a tray inside a
  tray drops its fill, never a second border weight.
- A section is: title (`--type-section`) → optional one-paragraph description
  (`--type-body`, `--text-meta`) → content → optional note (`--type-meta`,
  `--text-meta`). The description explains what the section decides; the note
  carries consequences and caveats.
- **Order by importance and expected use.** Put first what a typical install
  needs most often or what matters most, and last what most installs set once or
  never: left to right for tabs, header actions, toolbars and pill rows, top to
  bottom for sections, cards and sidebar entries. Decide from what the control does
  on desktop, browser, Docker and phone installs, not from counts: the UI never
  measures use and never reorders itself. Related sections stay together and move
  as a group. The exceptions: the default tab is first; a setup flow keeps its
  order (connect an account, choose models, assign agents); an ordered scale and an
  owner-defined order (Widgets) are never re-sorted; destructive actions and About
  come last, while an emergency stop (Panic) stays where people reach for it; Send
  and Save keep their usual place. A familiar location moves only for a clear gain,
  and a new section takes the place this rule gives it, not the top because it is
  new.
- Subsections inside a section use a `--type-body` semibold heading and stay
  visually grouped with their own rows, their own add action in the head
  (List editors, below). A heading that floats equidistant between two groups
  belongs to neither.
- **A disclosure shows that it opens.** Its trigger carries a visible open/closed
  marker — the native triangle, the `▸`/`▾` pair, a caret or button/card chrome —
  and reads as a control: a summary beside a help line at the same size takes
  control ink (`--text-primary`) and its own line; one that already reads as a
  control may stay in meta ink. Overriding `display` on a summary drops the
  native marker, so draw it. What opens has one coherent boundary tied to its
  trigger: what a bar discloses unfolds from the bar's bottom edge as its Shade,
  and opening moves no neighbouring target. **Floating chrome stays readable:** over scrolling
  content, closed or open, content never shows through its text, and state tints
  sit on that backing. Text controls in one floating group share height and
  baseline; width, tone and icon shape follow their role. The group reflows by
  its own container's width without losing labels, targets or focus.
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
- **A markdown heading inside chat is never a page title.** In a full rich
  answer (§1: answers, the Project completion mirror, `markdown: true` System
  rows) it is a section of the answer and never smaller than the text it
  opens: the §1 reading ladder — `#` = `##`, then `###`, then deeper levels
  at the reading size — semibold, in the bubble's own ink, with clearly more
  space above than below. The compact renderer's surfaces keep their density
  (§2): in a Skill Review report every level is a `--type-body` semibold
  label, and in a task card's timeline a heading renders inline, without
  block margins, at its row's own size, with a copyable line break before
  the following paragraph. Other rich-content cards (the question card, the
  update letter) keep `--type-body` semibold labels. Bold text in chat
  Markdown and rich-content cards is semibold, so a bold lead sentence never
  outweighs a heading. The page-size `md-h1` belongs to non-chat surfaces
  only.
- **Rendered Markdown keeps its author's structure and text.**
  - A table stays a grid and fills its column. Cells wrap between words and
    align to the top; the author's GFM column alignment holds (numbers right,
    centred columns centred) with tabular figures; only a link, hash or code
    span may break anywhere. In a full rich answer it reads at the answer's
    text size; compact surfaces and cards keep `--type-body`. A table in
    either renderer scrolls sideways only when its columns cannot fit even
    wrapped, and only then is it a named keyboard region (`Scrollable
    table`) whose sides that still hide columns fade. No stacked row cards,
    no guessing a table's purpose from its words, no smaller text to make it
    fit.
  - A thematic break is a `--surface-border` hairline with section spacing.
    Inline code keeps the ink of the text around it on its neutral pill.
  - Code is the author's literal text: a fence or code span shows and
    copies exactly what was written (`&amp;`, `<div>`, `**stars**`
    included), and no formatting rule reaches inside it. A rich answer
    reads code as its Markdown parser does. The compact renderer keeps the
    promise for its own narrower grammar only: a fence opened by three
    backticks, an optional label made of ASCII letters, digits, underscores
    and `+ # . -` (at least one letter, digit or underscore; `c++`, `c#` and
    `objective-c` included), optional trailing spaces or tabs, and LF or CRLF,
    closed by the next three backticks; and a code span of
    one or two backticks around text on one line. Other code forms (info
    attributes after a space, a span across lines, a longer backtick run,
    a `~~~` or indented block) are outside that
    grammar, and the compact renderer does not promise them literal. Rich
    prose reads one stored entity layer (`&amp;` shows `&`); an entity never
    creates structure, and raw HTML stays literal text.
  - A Markdown image is never loaded. It stays a visible reference,
    `Image: <alt>`. In a rich answer it is a link through the chat URL policy
    when its address is allowed, plain text otherwise, and inside an authored
    link that link's own text; the compact renderer links it like its other
    links (a refused address becomes their `#` placeholder). Chat media sent
    through the media path is unaffected.
- **A narrow chat column gives every bubble its full width.** A column of
  620px or less — a phone or a narrow Project panel alike — drops the
  wide-screen gutters; the column, not the viewport, decides.
- **A task card's summary outranks its details.** The latest-activity line is
  `--type-body`; collapsed timeline rows are a dense log at `--type-meta` in
  `--text-secondary`; an expanded row returns to `--type-body` in
  `--text-primary`. Details never render larger than the summary above them;
  an inline label inside a row is semibold at the row's own size.
- **A nested child card is subordinate to its root.** Its compact identity row
  shows status, role and `Details`; a short task id disambiguates siblings. Its
  compact line names the executor with only its own model evidence (observed;
  `requested …`; unconfirmed absent), else the agent model. Coordinator and
  provenance stay in details; effort stays visible as the leaf's dispatch choice,
  not a provider echo. Its chip uses the child's own frames: the applied level (`Effort High`),
  the asked level when it differed (`Effort High (asked Ultra)`), then its
  source: a pin glyph before a pinned level, `· model name` for a level the
  model name carries, `(Cyber Pro)` under Cyber Pro, nothing for the owner's
  range. A frame without a level paints no chip; an older frame is never
  back-filled. A child keeps one useful activity line visible, a root up to
  three; empty activity reserves no band, and a duplicate title is not activity.
  Full narration and Reviews expand independently. The root keeps primary title
  ink at weight 500, children secondary ink at 400. Nested frames preserve real
  ancestry; their opaque secondary surface avoids accumulating translucent white
  tints at greater depth.
- **An executor label names the helper and states only exceptions.** The
  ordinary case is the bare name — an assigned route beside the card's own
  `Working` chip, or every settled run ok; it names the selected helper and is
  not proof of current computation. A progress actor adds `last update`; a
  refused pin, no run, runs not settled, failed or unverified runs, unreadable
  evidence and an undisclosed outcome split qualify it. One priority phrase
  stays visible; the detail names the other exceptions. Terminal records without
  run evidence remain explicitly unknown; no promise of a later receipt replaces
  it. Counts, the model and its source (requested, observed or `model unconfirmed`), spend
  and access form one detail sentence that the expanded card prints as text,
  never only a title. Settled observed models are separate historical facts,
  not a claim that their union is the current actor. Missing identity stays
  unconfirmed; marks and configured routes never manufacture execution.

A completion notice distinguishes the task's recorded end from the time the
notification was added, both as absolute local dates outside the answer's
copyable body; an unknown historical end time says so explicitly. Inside the
room, the task card's saved end line keeps its notification time and adds the
same note whenever the recorded end falls in another minute or is unknown;
minutes compare as instants, and two that read alike on the local clock (a
repeated daylight-saving hour) carry zone names. Lines saved before the host
recorded end times stay as they were. Delivery keeps its present place in Main;
an older Failed remains that task's result even after a different task succeeds.

### Document reading

Markdown and plain-text chat files, Ouroboros's or the owner's, open with
`Read` over the conversation (a full sheet on a phone). Photos (also `View`
in their `•••`), owner pictures and delivered PNG, JPEG, GIF, WebP or AVIF
files (`View`) open there too. Other files, SVG included, keep the Open /
Download dialog; the reader keeps both, plus image Copy. Close, Escape or the
backdrop return focus to the opener at the same place. Leaving that chat's
screen (another page, a Project room over Main, a notification) closes the
reader; composer files stay.

- Text uses inline bytes, a delivered artifact or current stored owner-upload
  bytes; uploads are not rehashed per read. Images use their tile's source.
  Text errors explain known refusals. Image errors hide HTTP reasons: URL
  images offer Retry even for 404; failed inline images offer Download.
- The header adds no message. Name/photo caption, type and size stay untranslated;
  controls translate. Titles preserve line breaks and wrap whole in a bounded
  keyboard scroller; actions wrap before squeezing the name.
- Images open at `Fit` (whole, never enlarged); `Actual size` scrolls their pixels.
- Markdown follows the reading size and heading ladder of a rich answer (§1).
  Single newlines inside a paragraph are soft; explicit hard breaks stay.
  Raw HTML and SVG stay literal, images stay references and never load, and
  relative links read as words. `Source` shows decoded text unformatted,
  dropping a leading byte-order mark, not a byte view. Nothing is editable;
  all text is selectable.
- Plain text reads monospaced and wraps; only tables and code blocks scroll
  sideways, inside themselves.
- Text reads at most the first 1 MiB. Longer files show that prefix, noting how
  much and that Download has the whole; a character cut by the limit is withheld,
  not called damage. Non-UTF-8 bytes show as � with a note; NUL bytes refuse text
  display. Reads and images unfinished after 30 seconds stop with Retry;
  nothing retries by itself.

### History edges

An initial read shows one disabled `Loading saved history…` pill at the feed edge,
not a second loading sentence in the header. Task activity remains separate.
Failure and approximate-restoration notices are wrapping text in the room's header
chrome: the Project status bar, or Main's floating note on its float-sheet backing (§5).

Main and every Project open at their newest message, regardless of age or other
rooms' traffic. Returning in the same session does too, including Main after
closing a Project (owner decisions 2026-07-10 and 2026-10-05). Only explicit
navigation lands elsewhere: a question from its Main mirror, an exact quoted message, or a row of the room's
work status. Its landing reads the bounded page around the target as the room's history
from there on: `Load more history` continues older from it and `Scroll to latest
message` returns to the present. Showing the window again keeps the reader in place.

A room's history is its own. A page is counted in the room's messages, never in
bytes of other rooms' traffic: the newest page holds the room's newest messages,
and `Load more history` reads the next older messages of the same room, only
ever older ones (owner decisions 2026-09-14 and 2026-10-05). It shows only while
older history exists, and one press keeps reading until messages land or the
room's beginning is reached, so no press is empty. A positive scroll gesture at
the reading edge loads the same way. Messages arriving while the reader is in
older history never move where the next press goes on: it reads above what is
being read, and anything a long absence left unloaded below is read by the
return to the present. Narration (task progress) keeps loading
alongside the conversation, but only the conversation keeps the control: once the
room's beginning is reached it leaves, even if older narration of the oldest
cards remains unread. A page around work that only narrated pages that narration.
The floating `Scroll to latest message` is the one return to the present; when
the present is already loaded it moves there and follows without a read.

A failed history read offers Retry; a failed read of the present by ↓ leaves the
view in place with the same Retry. A failed full-output read keeps the line's
capped preview and offers no Retry of its own; collapsing and expanding the line
asks again. Reading on, even inside a bounded box, supersedes a ↓ or question
still loading. A wheel, swipe or key over a bounded box moves that box, not the
conversation, until the box reaches its edge. A scrollbar drag follows new
replies only when released at the live edge; a wheel, swipe or key reading down
follows once its scrolling ends there. Expanding a line or receiving its full
output leaves it in place; only a newer line moves a card timeline to its end.
New replies remain below in the same live conversation without moving the passage.

When loaded fragments are disconnected or their coverage is uncertain (reading
far back releases the newest pages from memory while replies keep arriving), a
toast above the composer announces once per document opening `Some saved history is not loaded.
Shown messages may have gaps.`, and the same readable explanation stays at the
in-feed `Load more history` edge, including beside a read failure or approximate position (owner 21.10).
The notice can be closed; hover and focus pause its six-second dismissal. A failed read stays in header chrome.
Bytes written after a read found a source empty count as missing until a later
read delivers them from its start. Mixed task cards keep one node and use this
general note: dates, common row IDs and an exhausted cursor cannot establish a
separator or full coverage. `Beginning of saved history` requires the room's
conversation to be delivered completely. Retained origins say `Saved project
context`; a matching canonical source row adopts that node and removes the label.
The context itself certifies no archive coverage. When a window is shown again
and the reader's exact place is gone, the view falls back to its card, then the
previous clamped position, and the same persistent note explains the
approximation until explicit navigation clears it. Visible rows and card lines,
selection, focus and expanded Reviews keep their actual nodes during
reconciliation.

### Room work status

Each room's bar has one status, right after its title: the short word in its
status ink, then one mark, as on the sidebar's rows; no frame at rest, a neutral
tint on hover and while open, a caret when it opens. A work is one request with
its Continue episodes and whatever of them still lives (helpers, delegated
runs still working, post-answer upkeep); the server gives each work and each room
one status, and every surface (bar, Shade, sidebar, cards, tray) words that
one status without deciding its own; Activity's rows keep their queue words, said
as the chip says them (never a raw phase code), and take the work's controls. A
card without a projected work (a child, Main's owner-action message, a replayed
card) takes its shape from its census phase through one table pinned to the
server's `mark_of`, and the owner's move only from proved facts (`can_resume`,
`requires_owner_action`, its own question). The room says its busiest work:
working, then queued, waiting, paused, and its mark is the server's status mark
(Status marks): moving dots while anything in it runs (a paused work's finishing review too), still dots while it queues or waits on
something that ends by itself, two bars while paused, a faint ring when
unconfirmed. While something waits for the owner the amber dot replaces that shape
beside the room's word; a Pause is not the owner's turn. At rest `Online` keeps its
green dot, a status sentence's own. Only what the client alone knows lies over the
server's mark: `Reconnecting…` (a ring), the owner's own unconfirmed send and the
owner's own Stop press (moving), and `Starting…` with the boot phases that refine it
(no mark). The transcript's typing bubble shows only while the room's work moves and
no card speaks for it. The title
ellipsizes first, and a new word never moves a neighbour or resizes the bar.
When there is work to show, pressing the status, or ↓/↑ on it, unfolds that
room's work and only that room's as one Shade under the bar, the same in every
room: one row per work, the owner's unconfirmed sends, and unfinished cards an
incomplete read has not confirmed (shown unknown, without controls). A Project
with nothing else keeps its latest represented root card. A row names its work (its
coined name or the request's title, never `Continue: …`, else the owner's
message) in two lines that keyboard focus or open details show whole, with the
state under the name and its quiet line beside it, the row's mark before it (an
outcome is a dot in its tone: Done and Done with warnings green, Stopped neutral,
Failed red). The word is the current episode's alone; the line says its reason, then
what still runs beside it (helpers, external runs, review work beside a task that is not
running, such as the review a Pause lets finish, a run nobody can confirm); a work card's chip says the same word with the same line
beside it, and the sidebar's name says both. An outcome stays beside `Finalizing…`; `Paused` stays until Resume, its
line saying who or what paused it (the owner's press with its time, Ouroboros in its
own sentence, nobody when a press declared no one);
`Waiting` says what it waits for and goes on by itself when that happens;
helpers an earlier attempt left running read as `Working`, with their count; an
access wait is not an answer wait, and unknown stays unknown. Under the state a
live work shows its own focus line, what Ouroboros says it is doing (stage ·
done · left, written by it through `update_focus`), quieter and at most two
lines; a work without one shows none, and a finished row drops it.
The name goes to the work's message wherever
it lies, finished work included: its card or the owner's message when loaded, else
the room reads the page of its history around the work's first message there and
lands on it, the list staying open while it looks. Before that a row claims nothing
about where its message is; when the room's history holds none (`No message yet.`,
or `No message saved in this chat.` once ended), belongs to another room, or cannot
be checked, its details say so. `View question`
goes to the exact waited question and answers nothing. A row offers exactly the
controls the server lists for its work, each at its exact task, on the shared
task controls: Stop ends the whole work (every episode still running, the newest
first), `Stop earlier attempt` only what earlier attempts left running, and
Pause, Resume, Hurry and Wrap up act on the current episode. A control its door
would refuse is never offered, so a Continue waiting for its earlier attempt
offers no Resume. While
open, order and focus hold, and work that ends keeps its row and outcome, without
controls, until close. Escape, an outside press or leaving closes it; `Back to
reading` returns to the passage left, reading its page again if it was let go, or
to the present the reader was following. Navigation never changes the next message's
recipient, opens another pane or manufactures activity.

### List editors

A list editor is any section where the owner adds and edits entries in place:
the Available subagents roster, MCP servers, custom keys.

- A section-level add action acts from its group's header (§6). A list editor's
  new entry appears at the end of its own group, is scrolled into view (the
  shortest distance, without animation) and takes the caret in its first field.
  A button that stays in view while the entry it made is born off-screen has not
  finished its job. Remove hands the caret the same way to the entry now in its
  place (the next, else the previous), or to the add action once the group is
  empty, never to the page.
- A freshly added entry is an invitation, not an error. Where a list editor
  validates in the browser (today the Available subagents roster), the entry
  shows a neutral hint in its own meta line until the owner tries to save; the
  error then names the entry and stands beside it (the entry tinted with the
  status pair, never dimmed), with the section-level line as the summary. A save
  attempt judges the entries that existed then; one added afterwards is an
  invitation again.
- A per-entry on/off switch is a native `.ui-checkbox` leading that entry's
  head, before its title, with its own accessible name and pointer target. It
  is a draft like every other field — the section's Save is the one writer, and
  no entry saves on its own click. An entry switched off is not dimmed, locked
  or removed: it keeps every control editable and its own status reading, and
  the switch never merges into the availability axis beside it.
- A multi-field card (an MCP server) follows the add-and-reveal rule without
  adopting the §6 row anatomy.

### Reviews inside task cards

Real tasks and real subagents are cards. Reviews are a subsection of the
exact real task that owns their presentation. Harness and neutral API marks
identify the delivery channel alongside explicit execution evidence; they are
not child-task cards and never prove execution by themselves.

- A collapsed task card, and a collapsed nested child card the same way, shows
  only a quiet `Reviews N` count, optionally with an active count, at the start
  of its one action row under the metadata line (its controls right-aligned on
  that row): no aggregate pass/fail alert, no synthesized verdict, no review
  dollars.
- Expanding `Reviews` reveals one row per currently admitted review group
  (`Skill review`, `Plan review`, `Review records`, or a read-only `Task
  acceptance` group for a panel stored before the result review). Expanding a group
  reveals its ordered attempt rows. Group state and verdict remain
  domain-specific; one blocker never recolours the whole task card.
- Start progress labels the frozen model/route/profile as requested; settlement
  reports that same slot's observed execution or says it was not reported. An API
  model sent in a request is not an independently observed provider label, and
  duplicate model slots remain distinct. No global last-run identity fills a gap.
- Disclosure is user-owned. Review results, retries, failures, terminal task
  state, reconnect, and lazy-detail loading update content in place but never
  open or close the task, Reviews section, or group.
- The Plan group is the plan record (nobody votes, so it carries no verdict
  word): its line reads `N objections may hold` (an owner-marked reviewer's
  objection, warning tone), `N open objections` (only added critics', warning
  tone), `awaiting answers` (working tone while its task runs), `no open
  objection` (done tone) or `not asked yet`; `objections not projected` in the
  neutral tone when the detail lacks the host's fold, which the panel never
  re-derives. The revision row shows the latest revision's goal, `done_when`
  and earlier revisions' claims; each ask row reads `k of n answered` while
  replies are owed and lists each participant (`added — never holds`,
  `awaiting`, `not sent — <reason>`) and its objections (`(not adjudicated)`,
  `(never holds)`). A record of the earlier schema keeps its wave rendering
  below.
- An earlier-schema plan wave whose reviewers may still answer reads as work in
  progress only while its own task runs: the working tone,
  `in progress · k of n answered` where the verdict token would sit, and each
  awaited reviewer listed as awaiting, not unavailable, under no verdict word;
  once the task has ended the same wave reads `no verdict · k of n answered` in
  the neutral tone. A reviewer whose window expired is unresolved. A settled
  wave whose reviewers were too few for a verdict reads
  `no verdict · k of n answered · m unavailable` in the neutral tone: the
  verdict word `DEGRADED` is the host's placeholder and never paints.
- A task-acceptance panel stored before the result review whose reviewers may
  still answer never reads as a bare verdict token. While its own task runs it is work in progress: the working
  tone and `in progress · k of n answered`, or `PASS so far · k of n answered`
  once the quorum is met; once the task has ended it reads
  `no verdict · k of n answered` in the neutral tone, or
  `PASS · k of n answered`, until a late settlement replaces it. A reviewer FAIL
  keeps the error tone and reads `FAIL · k of n answered` in both.
- On a plan wave and a stored acceptance panel alike, a slot neither answered nor
  awaited (a settled failure, an expired window, lost custody, a refusal) adds
  `· m unavailable` and keeps the warning tone beside the awaited slots; a
  settled plan wave with no awaited slot reads its `no verdict` line in the
  neutral tone (a stored acceptance panel with none reads the same line but keeps the
  warning tone, its stored `DEGRADED` staying in the attempt detail), and each
  unavailable plan reviewer row names the model and quotes the engine's reported
  sentence when one exists, the failure code staying in the task detail and
  Logs.
- A plan wave the mind ordered weaker than the owner's effort setting says so in
  its attempt detail as one keyed line naming each seat, the ordered effort and
  the setting (`Reviewers ordered weaker than your setting: s1 low (setting
  xhigh)`), at desktop and phone width alike; the verdict token is never
  recoloured for it. A seat that did not answer a same-spec cycle adds `· did
  not answer; its earlier finding is still listed` to its unavailable row (a
  never-sent seat: `· not sent; its earlier finding is still listed`).
- An awaited or unresolved reviewer row adds `· since HH:MM` in the viewer's
  local 24-hour clock, prefixed with the short date when the wait began on an
  earlier day, only where the host recorded the moment it sent that reviewer's
  request; a time is never inferred.
- A review record whose wave recorded the review pool says it in words at the
  top of its attempt detail: one panel line (the strictness, who chose the
  panel — Ouroboros, the host or you — `k of n ticked reviewers` and any
  advisers, `not counted`), the reason given or `widened to every ticked
  reviewer (Blocking)`, each seat not reached with its cause, and the wave
  total over the priced seats, naming those on a subscription or of unknown
  cost. Each reviewer line leads with `counted` or `adviser, not counted` and
  its own cost (an amount, `subscription` or `cost unknown`, never $0 for a
  gap). A seat still owed an answer reads `awaiting an answer`, never `not
  reached`; a preflight record keeps its own one-row line. A record written
  before these facts reads as it always did.
- A panel that settled after its task ended stays one attempt row of its group,
  labelled as settled after the task ended; its note (which verdict, which
  revision, whether a reviewer's outcome is still unknown) is host-composed and
  printed verbatim, leading the attempt detail.
- Stable keyed rows are reconciled in place: a routine update preserves the
  exact lazy-detail node, focused descendant and its reading position. Expanded
  groups state exact aggregate accounting when projected, attempt detail exact
  accounting when the domain can prove it, and otherwise each says
  `Cost unavailable`; collapsed rows never show dollars.
- Harness marks are monochrome `currentColor` vectors with adjacent visible
  text. They carry identity only, remain neutral across status states, and use a
  generic text-preserving fallback for unknown harnesses; direct API is shown
  neutrally as `API`.
- Vector provenance: Claude, Cursor, and OpenCode paths come from Simple Icons;
  Codex/OpenAI comes from SVGL. Product names and marks remain the property of
  their owners.

### Quiz card

The owner quiz card (`web/modules/chat_decision.js`, `.chat-quiz-*` in
`web/style.css`) is a chat-delivered decision surface. Optional clarification is
fire-and-continue: the task states its assumption. Required waiting keeps the
same card and says explicitly that the task awaits the owner, with Stop and
existing task deadlines still effective; an optional bound on that wait resumes
the task with a host notice and leaves the card open, the notice saying what the
card does (with a stated assumption the task proceeds under it; without one, no
answer is explicitly not consent). Silence is never an answer on either surface.
A card outlives its asking task: after the task finishes, the owner can still
answer, and the answer arrives as their own message in that chat. After
settlement, status and the owner's recorded answer keep both forms readable.
Anatomy, top to bottom:

1. **Head** — one 28px row in every room: neutral `Question` chip (`--type-meta`,
   neutral pair), the status beside it as a 7 px status mark + text, amber while
   the task waits for the answer. The row never wraps (a narrow card's long status
   takes two lines in place). The voice line names who asks and a Main copy's
   Project; private Presence questions and reports add the linked place/thread
   source beside the name, outside authored text. The lifecycle family leads with the one
   word that answers "is there an unanswered question for me?":
   `Waiting for your answer` needs positive wait evidence (the task's live wait
   record, or the original required flag before any record exists); a resumed
   wait — owner input or other mail woke the task, or the bound closed — reads
   `Unanswered · task continued`; an open question without any wait evidence
   reads `Unanswered`; `Unanswered · task finished` keeps the neutral dot and
   answerability (a late answer arrives as the owner's message); `You answered` uses the recorded-answer word and mark
   roles: owner blue in dark, the existing meta/ok treatment in light; unknown delivery retains its owner-action mark;
   `Replaced by a newer question` and `Question closed` use the disabled dot; an unreadable source
   reads `Status unavailable`, never an invented invitation. No answer-deadline
   countdown: task completion closes its mailbox, not the question's answerability.
   Pressing immediately shows `Sending…` and the brand-marked choice, keeping the
   full form and focus, the head's mark neutral until the answer is confirmed; inert controls are faded in light (in
   dark they keep full ink beside `Sending…`) and submitted fields become
   read-only. No success before canonical confirmation. `Not confirmed` means unknown,
   not failed: only then, under the answer, show Forget unconfirmed local answer
   (quiet, first), Check status and Retry same answer. Recovery first reads exact-quiz
   truth; explicit retry sends identical
   bytes. Forget cannot cancel or retract a server answer; answered-with-delivery-owed
   remains recorded.
   While the card takes an answer the head ends with one quiet × (`Close question
   without answering`): one tap closes it, no reason to type, and it waits while an
   answer is typed, so a mis-tap never discards the owner's words.
2. **Question** — the one primary thing, by position and ink: readable
   `--type-body` text in `--text-primary`, regular weight. Its emphasis is the
   asker's own (authored headings and `**…**` are semibold), so a question of
   several lines keeps a title instead of reading as one bold block. Its
   paragraphs, lists, checklists, tables and code keep the shared rich-content
   gutter, rhythm and bounded code scrolling. The card never infers a title from
   the first line or rewrites authored Markdown to fit, and the question has no
   quiz-specific length cap. The bubble ends on its time alone: the host's
   sentence (`host_facts`: asking task, run start, the owner's last message) is
   not drawn.
3. **Stake** — optional one-liner (`At stake: …`), `--type-meta`, `--text-meta`.
4. **Options** — zero to six real owner actions: elevated cards (the capsule surface
   on the rest shadow with the object radius; a choice, never a field), the label in
   `--text-primary` at 600; an optional per-option detail steps down to meta ink inside
   the card. Hover and press change the card's colour. After settlement unchosen options lie flat and remain
   readable. The chosen answer uses the chosen-answer roles: sapphire/white with its rest lift in dark, the existing
   green/meta flat treatment in light. Its accessible name identifies it; dark also shows `Your choice`.
   Recommendation is separate from the owner's answer. Options are capped by the shared Python↔JS constant
   (`MAX_QUIZ_OPTIONS`); with none, the free answer is the whole answer.
5. **Free answer** — while the card is open, one always-visible frame (the field's
   hairline and focus idiom) holds the words, named `Your answer`, and at its trailing
   bottom edge the `Send my answer` crimson capsule at the composer's height, enabled
   and standing only once something is typed; a narrow card gives Send its own line in
   the frame. It is a message field: its words grow to one cap, then scroll; Enter sends and
   Shift+Enter breaks the line (§3 "Controls and editable choices"). No option
   ever has to be the least wrong one: the text rides with an option click as
   the owner's remark, or goes alone as the owner's own answer. It uses the
   card's own ink and surface tokens (never the legacy chat input), is capped
   by the shared Python↔JS constant (`MAX_DECISION_COMMENT`), and disappears
   the moment the card settles.
   Origin-local drafts and frozen attempts persist across closure only with a
   persistent browser profile (older private WebViews erase them: Appearance above).
   Restore reads, never resends; storage failure warns to keep/copy words.
   Emptying the field releases its draft and a refusal note about it, never a sent
   or unconfirmed answer; nothing else clears drafts. Canonical
   settlement clears the copy; winning bytes appear literally as `Owner's answer: …`
   under the options, beside the highlighted choice or as the whole answer if none,
   retaining line breaks (`--type-body`/`--text-primary`).
6. **Assumption or waiting** — the signature line (`Continuing meanwhile: …`
   for optional clarification, an explicit waiting statement for required input),
   `--type-meta`, `--text-meta`, parted by space. While the card is
   open it names the default path. The optional assumption remains after settlement
   as the record of work continued meanwhile. Required waiting copy disappears
   when the card settles; the status and recorded owner answer remain.

Inside a bubble the card adds no box: the bubble is its object. Its × is a quiet
control, without a body until hover. The card uses tokens for every
type size and colour, including pill radius and option gap. Every focusable control shares the 2px `--focus-accent-border` ring
with 2px offset; the answer's frame wears the field idiom instead (§3 Focus). Component min/max widths keep local literals, as elsewhere in chat.

**Project question mirror.** Main is the list of what needs the owner; nothing waits in a section of its own. A Project question's Main copy is the Project's own `buildQuizCard` card, whole as anatomized above, in an assistant bubble at its ask time in the feed; the Project reference in its voice line opens the exact original, and answers, materials and recovery controls work in Main. It uses the shared ring and `--project` tints; names ellipsize (full titles); the head stays the room's own row. History, live pointers and the census share one task/quiz card and lifecycle wording; the canonical current questions only confirm or settle copies the feed holds, never mount them, and an incomplete read withholds only what its typed gaps name. Optional, resumed and terminal-author questions stay answerable until answered or closed; no display/history cap drops them. Replaced questions stay read-only. The owner closes from the card; a model closes its own or a host-proven continued task line's question, or a finished task's, always with its own reason. Closure records a reason, never an answer/approval. A late answer left delivery-unknown by a restart offers Remove from pending (no resend, never marked delivered). Publication destination stays fixed across Project conversion; current room membership controls display. Explicit materials use registered task artifacts and the existing reader.

A task that cannot continue without the owner is one System message in the feed, stamped when first seen: the task, one sentence, its own controls (revision-bound model access, Resume/Stop, children's included; activity counts each work once, its helpers within it). Sign-in, unobservable automatic access, disabled automatic continuation, budget pauses and Resume-only holds qualify; viable automatic account/quota recovery, queues, self-waking sleep, background scope verification and own Pause stay quiet. Deduplicate the same quiz/wait identity, not all waits on a task; the message leaves when a complete census drops the row. Missing action evidence stays unavailable; visibility never authorizes login, budget changes or automatic Resume.

One amber pill among the composer's pills (their row, height and type, at its end; `--status-warn-*`) counts the unanswered questions and owner actions — `1 question`, `3 questions`, `1 action`, `4 waiting` — and shows only while one of them is off screen; a narrow row keeps the glyph and number. Each press shows the next, oldest first, cycling: the card head lands under the chat header and the question text takes focus, never the answer field, so no touch keyboard opens. A question outside the loaded history mounts at its own date under the existing history note; there is no history seek.

A confirmed answer/closure shows its result for five seconds, then retires only the Main copy through message/media disposal without moving the viewport; the Project card stays. Pending canonical delivery keeps the read-only form and a compact pending fact. The interval starts once, including simultaneous form/result delivery. Focus remains during it, then moves to the next Main question or keyboard owner's composer without summoning a touch keyboard. History, reconnect and stale snapshots cannot resurrect settled forms; a settled pointer never mounts unless its delivery is owed. A newly accepted question closed before first exposure publishes only its closed state; an old answer without explicit delivery debt creates no new owner turn.

Unavailable canonical refresh retains text, references and drafts but disables affected controls. Missing, failed or wrong-Project reads say `Status unavailable`; absent form text says `Open the original question for its text.` The chip opens the original; owned refresh retries.

**Work portal.** A work living in a Project room keeps its one Main card at the
request's place, whoever moved it (Turn into project, the mind's move, a
binding); a Project start or transfer row only says where it went, never a
second entry. The card is the work's portal: status chip and quiet fact, title,
one latest line (two lines, cut on a word), the meta line (tool calls · money ·
updated), one action row `[Reply] [Resume] [Stop…]` with one primary, and the
Project reference as its full-width footer. Timeline, Reviews and children live
in the room; the summary is reading content, not a disclosure. The room's census
row supplies the present (its last line, tool calls, `in background` for
detached work); a missing fact leaves the card as it was, never a guessed line
or count. When the work ends the line goes and the chip states the outcome from
the final's typed outcome or the durable detail. An undurable Main receipt keeps
a dashed border and a plain warning; the binding holds. The owner's routing line
under his message steps aside while that work's portal stands in the feed; words
passed on to the work later keep theirs. A Continue is the same card: the
admitted successor takes the work's place and first name. Each episode has its
own name, time and past outcome; only the current one owns the chip and controls.
Expanded ordinary cards disclose earlier Files, timeline, Reviews and children
in collapsible sections using their original content, never duplicate cards.
A Project portal still reaches that content through its room.

**Project completion mirror.** Eligible managed/transferred model-authored finals reach Main as host-stamped Markdown in Ouroboros's voice; ordinary Project-native replies stay local. Long answers keep a seven-line footprint; only painted fading text extends two lines into footer whitespace, without moving Project/time/actions. Short answers have no fade; limitations/details keep the tail above their words. Complete text stays selectable; folded text has no keyboard stops. Project opens the complete answer and delivered documents; Main has no expansion or Files inventory. A nonclean final has one compact, wrapping limitation from the recorded outcome/review; a clean final has no status caption. Copy and Reply quote the answer alone. Ordinary document readers, quiz materials and task-card Files stay on their existing surfaces; retained availability is not delivery or owner receipt. Full reviews, logs and progress stay in their sources. Durable rows/wire frames keep System metadata and text; Reply uses the persisted visible answer without changing its address. Telegram status replies reference the work instead. Endings without an authored answer keep System voice: Main shows the work's name and true outcome, its cause, accessible recorded details and any labelled preserved output. The Project is named once by its reference; Copy, canonical history and Telegram retain full context. Older rows without these typed facts keep their stored text. A quiet scheduled run (it finished without a message) keeps its internal result and a terse `Done · no message` row in its own room; it creates no Main completion summary and sends no finish notification, while a host notice it carries stays visible. One logical identity joins retries/history and keeps one-ending-one-notification; external transport stays at-least-once.

History with no current execution or known outcome keeps its expandable content under `Outcome unavailable`, without a task chip, typing or Stop. Before complete live-source reconciliation, it is `Activity unconfirmed`. Positive current activity restores only its proven controls. A delivery warning may coexist with a preserved result-review PASS. Model metadata says `Last solve response`, naming the initial request only when the route changed.

History placement is separate from the task's outcome: ordinary
`Saving task history` and `Task history saved` appear only in task details and
Logs, and a storage failure adds `History storage problem` to the existing
collapsed-card metadata without changing Done/Failed/Stopped or creating a
chat message (successful retention clears it). Details and Logs name the
recorded failure reasons, repeated ones grouped with their counts; Logs Raw
keeps the complete source/failure facts, and a missing recorded reason is stated
explicitly. Existing task-detail hydration and replay refresh this projection.

### Living cards

A living card is an object Ouroboros made inside its own message (`show_card`), never a host decision: no status
chip, no amber, no answer field. The bubble is the object, so the card adds no box of its own: its title
(`--type-section`, 600) leads, then its body — components drawn exactly as on the Widgets board, a skill's widget, or
an app in a tray — and a foot that says `Ouroboros made this` in meta ink with the card's capsules on the right:
`Keep on Widgets` for a component card (then `Kept on Widgets` with `Undo`, or `Remove` when it was kept before; on
the Widgets page the kept card says `kept from the chat` and its menu carries `Remove`) and `Make it a skill` for an
app, which only puts a request into the draft. Until the body arrives, and whenever it cannot be read, the card shows
its plain words (the text Telegram gets) with one status line — `Loading the card…`, the failure with Retry, `This
card is no longer available.`, `This skill is off.` — never an empty box. A press on a card's control is the owner's:
it says `Sending…`, then `Sent ✓` (`Saved · the task reads it when it continues` until a paused task reads it) or
the honest failure, and a control meant once locks after it counted. An app's
frame reads the theme roles of the material it lies on (inside the dark Ouroboros voice, that voice's inks and
surfaces), at mount and at every theme change. A frame in
a message (an app, a skill's widget) is at least 120 px and at most 70 % of the window, with `Expand` / `Collapse`
beyond; at most three run at once, started near the view, and a far one rests as its words with `Open`. A running
frame has `Close`, which leaves it as its words with `Open` on this device until the owner opens it again; a frame
held after a start the page did not outlive rests the same way with its one line (Skills in the page). An app's
script error, unhandled rejection or policy refusal shows on the status line (`App error: …`). What an app hands
back lands as one line in the owner's draft: nothing in a card ever sends. Copy takes the plain words, Reply
and the row menu work as on any message, and a card that grows while the room follows its newest message keeps it
in view.

### Chat attachments

The owner's photos and files are part of the owner's message, shown the way
Ouroboros's own deliveries are: one message, one bubble, never one bubble per
file (owner choice 1C/2A/3A/4A/5A/6A with mobile first-class).

- **Order:** attachments sit above the caption inside the owner's bubble; a
  message with no words shows its attachments alone, never an empty text row.
- **Photos are whole on both sides.** One photo keeps its own ratio up to
  `--chat-photo-max-height`, never enlarged; a tiny, very narrow or very short
  one sits inside a `--chat-photo-min-box` box so its `•••` is never clipped
  (either side); several share the existing two-column grid with
  letterboxed (`contain`) tiles, one column in a narrow chat column. Ouroboros's
  own photo tiles letterbox the same way — no side crops an image.
- **Video and audio** play in the existing inline players; every other file is
  the existing file card (no PDF viewer). A preview the engine cannot show (HEIC
  in Chromium, an unsupported codec) becomes an ordinary card with Open/Download
  and "Preview unavailable"; a file deleted since it was sent, or whose size or
  mtime no longer matches the recorded stat witness, is the inert card
  "Unavailable". The witness is not a hash and nothing rehashes per view: a
  same-size replacement that restores the mtime is not detected and shows as the
  original.
- **Actions never depend on hover:** the photo `•••` and file cards are always
  visible and finger-sized under a coarse pointer; a long name wraps to two lines
  and the card's dialog shows it whole. Open/Save succeed only when the desktop
  bridge says so; a copied link is a fallback, never reported as opened.
  The photo's Copy to clipboard puts the image itself there, never an address,
  or says it could not (Open in new tab and Download stay).
- **Handing a file to a task adds nothing on success:** it arrives in the task's
  own message. A file the host could not attach is named in one wrapping line
  right under the owner's words ("Not attached to the task: …", from the routing
  receipt, live and on replay, never folded away in Main), and Telegram gets the
  same line. A new turn whose every file was refused is the System mark "Not
  started: the attachments could not be staged (N files)"; one that starts with
  only some of them says so in its own words.
- **The same view everywhere:** the sender's bubble, another tab and the replay
  after reload or restart render the one server view of each attachment. The
  stored text still names the files for the model; only the exact tail the web
  composer generated, on its own row (or text the row marks as host-written,
  `text_placeholder`, such as "(image attached)", a mark the echo carries too),
  is hidden from the caption: the owner's words, even those same ones, and a
  Telegram or skill caption are never rewritten. Every attachment renders,
  however many; only the text tail is bounded. A Project's start message copied
  from Main shows the attachments its row recorded when that row is outside the
  window. History without recorded attachments stays as it was; nothing is
  guessed from a file name.
- **Composer:** each staged image shows a small thumbnail beside its name; the
  field stays focusable (read-only, not disabled) while files upload, so a phone
  keeps its keyboard. `/restart` typed with staged files stays the Restart
  command (the files ride its row); any other words are a message. A sent
  message (words or files; a bare slash command is a command) is not yet saved:
  if the socket closes, or the host refuses the frame, before its "Input saved"
  echo, the bubble says "Not confirmed as saved" with "Send again", which
  resends the same frame and id (the host rejoins it if it was saved after all,
  never a second message), and "Discard", which forgets this tab's copy (never
  an upload). Nothing resends by itself. In dark, queued bubbles retain opaque readable ink; the queued
  words carry the state. Light retains its existing opacity. The frame is kept, with no count bound,
  until that id's saved echo or history row settles it (a row the running host
  has not yet dispatched keeps it until the host says which): a bubble the feed
  released or rebuilt keeps its frame, and a Project room holding one is hidden,
  not destroyed. The tab's sessionStorage keeps its words, id, routing and
  upload references (never file bytes) across a reload (the asset reload a
  reconnect forces included) and a room's teardown; after a reload the first
  history read settles it or shows the same doubt. A browser that refuses to
  keep it says so, and Send again then lasts until a reload; a kept copy it
  cannot read back is said too, never silently dropped. A saved row the host
  knows never reached the agent (its write failed before dispatch, in the
  running host) says "Saved, not delivered." and keeps Send again: the host
  hands that row over once. A kept message whose saved row a host process before
  a restart took says "Saved; delivery not confirmed." with Send as new, which
  only puts the words back in the composer as a new message: nothing is
  replayed. Rows nobody kept stay plain history ("Input saved").

### Replies

A reply names the row meant; Ouroboros decides what to do, routing nothing by it. Saved rows offer Reply beside
Copy on bubbles and in row menus, a work card in its row menu only; never on hover only. Reply stages one
strip, the composer field's first line (`Replying to`, quoted author, one line, ×; Esc cancels). Send
carries only the address; the host makes the snippet. Answerable questions use their own field.

One quote component serves bubbles, strips and relays: above the text, a 2 px
bar and tint in the quoted author's ink (the tint lies on the quote glass, so inside a
dark voice it reads as a shade of graphite cooled or warmed by the quoted author, not as
a second bubble; inside the dark Ouroboros voice a quote of Ouroboros lies on that voice's
capsule step, its author in the voice's readable word), one line cut on a word; a photo leads
with its thumbnail. Pressing a reply quote reveals the exact row, loading its
bounded page in this room when present, otherwise its available home. Containing
timelines open and scroll; the line is focused and highlighted below the header.
A card, portal or retained copy cannot substitute. A refused or failed read
keeps the view, and a reader following the newest message keeps following it;
its toast says why and offers Try again only when asking again can differ. Newer
gestures/navigation cancel it. Only the quoted row's stream is positioned: the
other stream (narration beside an owner message, or the reverse) stays unloaded
down that page chain until `Scroll to latest message`. Old sources read no newer
messages. Ouroboros quotes the message it answers only when it is not directly
above (its own card does not count).

A relay is a note one work leaves another (`tell_task`), written for that work, so
its room draws a note card, never an Ouroboros bubble (`web/modules/chat_relay_deck.js`):
the sender's room reference built by `project_reference.js` (Main is named without a
glyph; an unknown room keeps its plain name), beside `note to this work · time`.
The teller's line for the owner (`summary`; without it the note's first words in meta
ink) ends with one `more` control. It expands the note as written in a tray, its
Reply and Copy actions, and every quoted owner message together; `less` folds them.
Visible owner quotes keep their blue quote component, where/when and named files
(captionless files by name), clamped to three lines above the summary until expanded.
All consecutive notes, from any sender and with or without the owner's words, make
one deck in the work card's material: the latest card in front, the edges of up to
two behind, Show N notes fanning it out in place, Hide folding it. Other visible row
kinds end a run; hidden rows and typing stay transparent. A folded deck with quoted
owner messages in earlier cards carries the blue mark `Your words in N earlier
notes` (singular for one); N counts cards, not quotations. Fanning out reveals those
quotes, and a jump to a tucked row opens its deck before landing on it.
The owner chose this rule on 22 Oct 2026 (variant A): the blue mark now serves the
16 Oct decision that his words stay prominent, including when their cards are tucked.
Telegram receives only a relay that
passes his words, as plain text. Main keeps its one receipt line. The row menu (right-click, Shift+F10, ContextMenu on the focused
row, long press on touch) offers Reply, Copy, Open project and a work card's
Resume, Pause and Stop… exactly when the card does. Esc/outside press closes it
and returns focus; selected words keep the browser's menu. On touch the row sinks a
little while the finger holds and lifts over the dimmed feed while its menu is open.

On a phone a row that offers Reply can be swiped left to reply, as in Telegram: it
follows the finger, a reply mark at the room's right side fills as it nears the reply
point (56 px), and released past it the row replies exactly as its Reply does; it
settles back either way, past the point with a rubber band
(`web/modules/chat_row_swipe.js`). Rightwards stays the drawer's; a field, a slider,
selected words, anything that scrolls sideways and an open menu keep their finger.
A saved reply's registered actions (Read aloud, a skill's message action) come from one registry: icon
controls named for their action between Reply and Copy, Copy keeping its corner, and items after the row's own
doors in the row menu, in words. An action that is on now names its state (Stop while reading aloud) and wears
the crimson ink; a work card offers none.

### Skills in the page

A skill adds controls only where the host opens a place (the composer's "+"
menu and capsule row, a saved reply's actions), and the host draws them: the
skill's label verbatim, one glyph from the host set, the host's capsule or
icon control, never the skill's markup. Connector icons may supply reviewed SVG geometry, painted by the host in monochrome currentColor with a fallback glyph and an adjacent visible label. In "+", after Attach file, each skill
has a section under its name with its items in the owner's order; the host's
own items close the menu under Ouroboros's name: Add dictation… while no
engine can dictate here, and Make a button with Ouroboros…, which starts a request in
the draft for the owner to finish and send. A chip is one of the row's
capsules: the skill's name, then the value its source answers in the meta ink
(24 characters; the rest and its description on the chip's title), with a dot
when the value has a status tone. It is read on its own refresh (15 seconds at
least) only while a room shows it on a visible page; a source that fails says
"—" with the reason and is asked again later, less often. "+N", the narrowest
capsule, lists what folds; with no room even for it (a phone's row, which ☰
leads), each chip is listed in "+" under its skill's name. A skill's trusted page module works
only through its small context (its provider, a notice, words into the draft)
and never draws into the composer or the feed. A module that fails is off for
this page with one notice naming it; the rest of the page keeps working.

**A start the page did not outlive is not repeated by itself.** A page
module, a skill's module or route frame and an app card are code the core did
not write; on WebKit (the macOS app, iPhone, Telegram on iOS), Android WebView
and Firefox they share the page's one thread, so one that loops or blows up at
its start takes the whole page with it, and no menu inside that page can be
pressed. The host writes each start to a ledger on this device before the code
runs and clears it once the page outlived the start; whatever the next open finds
there is not started by itself. A held card rests as its words with Open, a held
widget as its facade with Start, both with the one line `Held back: the page
stopped while this was starting last time.`; held page modules are named in one
notice with Start it. The owner's press starts it (an app without the state it
saved, which may be the cause), and a new revision of the code starts as usual.
Code that stops the page later than a second after its start is not held; Close,
Stop or the skill's switch stops it. The way out of a page that froze is the
surface's own: quit or close it and open it again. Opened with `?hold=all`, a page holds every such item for that
load (`Held back: this page was opened with everything held.`), each released by
its own control. Ouroboros reads the fact in one System row in the chat that
owns the item.

### Voice controls

Dictation and Read aloud are host controls over engines that skills provide;
an engine is never chosen or replaced silently, and each one names where its
audio or text goes: On this device, On this computer, On your Ouroboros
server (its host), or Goes to the vendor it names. The composer has no mic
until a speech-to-text engine is chosen and can run here; until then "+"
offers Add dictation…, a sheet of the live engines, built-in Browser speech
(free; the skill's own words say which vendor hears the audio in each
browser; choosing it turns it on, or finds it in Browse while it is not
installed) and Whisper from OuroborosHub (its Skills entry when installed,
the Browse search when not). The mic is a quiet
finger-sized control between the field and Send; its chevron lists the
engines with their destinations, Get another engine… and Turn off dictation.
While recording, the slot holds a level dot and the time, Stop in the mic's
place wearing the crimson tone, and Cancel; the draft stays visible. Stop
transcribes; the words go in at the caret saved at Stop when the draft has
not changed since, otherwise a sheet above the composer holds them with
Insert and Copy. Dictation never sends. Esc, Cancel, the page's Panic, a
hidden room and the engine leaving end a recording; a chosen engine that
cannot run here is named with its reason in the chooser. Read aloud is a saved
reply's action (Replies): the reply's words with code announced, not read; one
reply at a time, Stop on the one being read; the page's Panic and leaving it
stop it. With no engine chosen it uses one on this device. The row menu's Read aloud
voice… lists Automatic (a voice on this device) and every voice, "On
this device" only when the browser marks it local, otherwise "Online voice".

Live Conversation keeps the glass wave, phone controls and collapsed transcript.
Talk occupies empty Send only when Calls (beta) is enabled and the server confirms
the selected engine is loaded and its credential grants are current. Text or a
file restores equal-width Send; empty Enter never calls. The dictation mic hides
during the call and returns with the draft intact. The glass keeps the composer's
fixed corner as it becomes a card; End remains visible in phone and keyboard
views. A model without vision hides Show and says, “This voice engine does not
accept images.” A hidden room's mark names the room and reopens it.

Calls (beta) settings default to Gemini, with calls off independently of keys.
Engine, model and voice choices come from loaded catalogs; an unavailable choice
is named, never replaced silently. Pause, interruptions, thinking and search
appear only where the selected model honours them, otherwise its catalog note
explains why. Instructions apply to the next call. The selected catalog names
where audio goes and its memory window per session, including the source and
whether it has been measured. Microphone, speaker, camera and echo/noise/gain
choices stay on this client and reach an active call immediately. Local device
tests release capture on leaving Settings and never take it from a live call.

Calls retain microphone, playback and socket when the room or document is hidden,
where the platform allows; dictation still stops on hide. End, Stop, Panic,
pagehide, device loss, unload and a successful Calls-off read end them. Failed
configuration reads keep the last good value; mount, focus, visible return and
successful Settings saves refresh it. There is no call time cap or separate
per-call reserve; the ordinary shared spending limit applies. The feature's
instructions keep the ordinary identity, memory and tools. It greets once per
call, never on reconnect. `set_call_languages` records recognition languages for
the owner or Presence binding and applies them at the next quiet renewal;
`end_call` is silent and ends only after the goodbye has played.

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

Narration leads; routine execution evidence stays compact; exceptions keep their
explanation and controls. The block's title, collapsed activity line and
timeline carry the turn's own narration (progress frames the model authored,
marked by the typed `narration` fact the worker stamps at that one producer; a
frame without it is legacy and read as narration) and whatever needs the owner's
eyes: a failed or timed-out step, a wait, a review, a child. Host notes on how
the turn runs (a checkpoint, a model fallback, a review verdict, a nudge) are
visible timeline rows that never claim the title or the collapsed line, so a
turn with only host notes keeps its coined or task name and an empty activity
line. A task that ends with a cause states it on the collapsed line in the
owner's words (the sentence of the durable row and the expanded body), so a
`Done with warnings` chip never stands beside an unrelated last action; a clean
ending keeps the last narration. Successful tool calls are not rows: they fold
into ONE evidence row per block (`N tool calls`, `N tool calls · M errors` once
a call failed) at the first call's position and time, patched in place; Expand
shows per-tool counts (`read_file ×3 · web_search`), and the phase (`calling`
while a tracked call runs, `warn` once one failed, `result` otherwise) shows in
ink, not words. A failed or timed-out call keeps its own diagnostic row and
counts once. A completed check command whose recorded criterion did not pass
keeps a separate expandable warning, “Check did not pass: …”, adding no call
error or task failure; its first sentence tells the check process from its
criterion, and its details name both and disclose bounded output. Live,
post-terminal and replay share that invocation-bound row; a later passing check
does not erase the earlier failure, and a comparison without a verdict is
unconfirmed, not a proven mismatch. On replay, recorded call times place the
evidence row before its diagnostics; a missing start permits only an
observed-time or legacy carrier fallback, not a reconstructed first call. Wait
end and operation settlement are independent: late success retires the
provisional timeout notice but keeps “wait ended” in the evidence row, and late
failure keeps its operation error, in either arrival order. A historical start
alone means outcome unknown, never Running or Failed.
`web/modules/chat_activity.js::toolEvidenceView` builds the row live and on
replay. Host invocation IDs join start/wait/settlement; legacy observations
without sufficient identity stay separate even with matching names and
arguments. Host metrics fill absent counts field by field; canonical
per-invocation evidence reconstructs later settlements on history/reconnect
without resurrecting frozen wait errors. Reads are bounded and carry coverage;
absent evidence is no proof of success. Start-only (live or legacy) and
incomplete replay evidence keep a known aggregate error; only complete
settlement evidence can replace an earlier wait error. Typed tool evidence after
task terminal updates counts and diagnostics on root and child cards, keeping
the terminal task phase and controls. The row keeps its live position; history
admits carrier evidence before summaries, progress or references choose their
presentation. Block presence is consistent across reload and reconnect. A moved
work lives in its Project, its Main card the work's portal ("Work portal").
`N notes` (timeline items, the evidence row among them) is in the `Details`
button's name, not its row. A threshold crossing names its bound and threshold without strategic
advice; distinct bounds and thresholds get distinct rows, and historical
producers without a named bound keep their old label.

Delegated executor activity keeps its own attributed voice in the root or child
timeline. Messages lead; thinking stays folded in its own details, and routine
technical events collapse into counts with per-event detail under Expand.
Problems, unread ranges and omitted previews stay visible with a full-source
reference. Preview limits include serialized size; they never erase the carrier
or its source. Overlapping reads deduplicate by sequence identity, retaining any
unseen tail's omission notice; equal words alone are not duplicates. A stream end
before the observed cursor states unresolved coverage. Live, reconnect and replay
use the same projection. Telegram reserves room for problems and incompleteness
beside its disclosed speech preview and keeps the complete task reference.

`web/modules/chat.js::blockHasWork` governs chrome from work, excluding open
attention and a bare outcome (owner 16.09: real work is a card, a greeting is
nothing). Reviews, children, evidence, narration or tool errors give managed
roots and direct turns the same card: a coined name or narration headline
(`Task activity` when finished without either; untitled while Working), status,
host-attested Stop, and `Turn into project` in Main unless already bound. Bound
direct turns route later rows to their Project like `move_to_project`. The foot
holds metadata and one action row: [Reply], Turn into project, Resume, Stop…,
one primary. Nested children stay inside their root, carry neither root control,
and convert with it. Attention-only blocks and non-Done endings without work
have their status and wait controls, no title placeholder or conversion; the
first work row adds the title.

The compact meta line names executor/model (roots: observed solve model, else
agent model), duration, cost, exceptions and running narration's `updated`.
One positive task-local tool-call count joins that line and remains after Done
(owner 22.10, A). Helpers are not aggregated. Rounds (usable solve responses)
and known zero calls stay in Details; missing counts stay absent. For footer
and Details counters, session nanny counts appear only in `Supervision`,
separate from unreported executor counts. Portals keep one count.
Neither rounds nor zero calls create work or a second footer row. Model
provenance, origin and a finished card's update time stay in Details. Money
there is ONE line whose wording carries its openness and never renders an
unknown as a number: `$0.00` only where priced rows evidenced that zero,
`up to $1.20` while the tracked subtotal is inexact, `Tracked: $1.20` when part
of the same scope has no price (`Tracked: up to $1.20` when that subtotal is
inexact too), and `Cost unknown` for unpriced rows with no tracked amount. An
empty, intact ledger has no money line; an empty view with an integrity gap
stays unknown; neither proves a free result. The chip's short word (`Sending`,
`Queued`) and `Tracked:` are explained in the card's details (exact phase;
`some steps have no price`), not a tooltip; real ` · ` text separates meta parts
for copies and assistive technology. The host's `_is_direct_chat` fact keeps its
host jobs (routing, census `kind`, Stop custody, terminal rows); the client
draws no chrome or status from it. A block that existed only for open attention leaves when that attention
closes: a wait-only block disappears when the wait resolves, and the resolved
episode's no-reopen ledger survives with the record, so a stale revision cannot
bring it back.

An addressing call (`start_task`, `move_to_project`, `tell_task`: the
routing-verb family `ouroboros/tool_capabilities.py::ROUTING_VERBS` owns) is
stamped by the host on its live tool-call frames (`routing_action`) and counted
in the task metrics (`routing_tool_calls`); its receipt is the typed routing
annotation on the owner's message. Such a call is a receipt row: it renders
inside a block that exists for other reasons but is never content the block
stands on, and the evidence row of a turn whose calls were all addressing calls,
without error, is a receipt row too (live from the stamped frames, on reload
from `routing_tool_calls`). So a turn that only addressed work («turn this into
a project») draws no block, live or on reload: the annotation on the owner
message and the managed root's own card are its whole record (owner decision
11.09). Successful local completion acts get the same receipt semantics through
the host's `completion_control` fact and `completion_tool_calls` aggregate,
keeping their actual call count without manufacturing work; failed completion
calls keep their error. A failed addressing call is an error row and therefore
content, as is any recorded tool error. A REFUSED addressing act is told where
the work lives and never in Ouroboros's voice (owner 16.09): the receipt line
states the cause in the owner's words, the failed call stays the error row
inside the block, and the tool result carries the typed reason (with the cause
and repair in `detail` where the producer holds one) so the model narrates; no
host bubble interrupts a narrating turn. When the host itself issued the act (a
Swarm message, a skill-card repair, a picker click), or a refused owner steer
carries no owner-message receipt, the host states the refusal as ONE typed
System row (`task_not_started`, `task_start_unconfirmed` for unconfirmed
admission, or `steer_not_delivered` for steering) in the issuing chat, keyed to
the named target. The Project start row is announced only once the task is
really queued. A refusal receipt with neither options nor a cause sentence reads
«Not routed», never «Choose a target». No client list of tool names decides
presence (`docs/development/02-naming-and-boundaries.md`, "an open default
behind a closed exception list"), and no client reading of a note's text decides
whether it is narration. A change that draws one row or line per event is judged
at a realistic burst size (a multi-call turn, collapsed and expanded, at desktop
and phone width), never at a two-event fixture.

### Subscription waits inside task cards

Quota exhaustion and a confirmed need to sign in again use the same component,
`model_wait.js` with `model_wait.css`, inside the turn's existing host: the task
card of a turn that has done work, the bare block of one that has not. Each
waiting role has its own row; model, account and reason are separate facts, and
the controls stay visible when the task's timeline is collapsed. Waiting carries
a quiet status, still dots and no computation animation, typing, activity counter or
invented progress; it is amber only when the wait needs the owner (the host's
`requires_owner_action`: sign-in, automatic continuation turned off, access that
cannot be observed, an account not proven, the cases of the owner-action System
message under Quiz card), never while it goes on by itself. Completed steps remain intact; a held worker slot and its
effect on the queue are stated only when the host reports that fact.

The quota row offers automatic continuation, initially enabled, and shows a
known reset time or an explicit unknown; an authentication row instead offers
the existing Accounts sign-in flow. Both can open Settings, retry explicitly, or
choose a replacement model/account in the shared model-role editor. The
replacement affects the named waiting role until its task or review operation
ends; an unchecked "Also save this role in Settings" checkbox separately
requests persistence. Fallback Local stays shared by its Settings group:
changing Local for one waiting fallback makes the replacement task-only, its
persistence checkbox cleared and disabled with an explanation while Apply
remains available. Model and account changes can still be saved when the shared
Local value is unchanged; permanent group-wide Local changes remain in Models.

A pool confirmed to contain both accounts requiring sign-in and accounts waiting
for quota says "Waiting for access" and names both causes. It keeps automatic
continuation and opens Accounts without selecting a profile or initiating login.
A dated pool refusal without a confirmed cause says "Waiting for an account",
retains the engine's reset forecast and keeps automatic continuation without
claiming quota exhaustion or guaranteed availability. It never offers sign-in.
Only confirmed quota pauses the execution clock; authentication and unconfirmed
availability waits consume it. Calendar deadlines remain fixed. Configured
fallback routes are tried before a wait card is shown. Work continuing on a
fallback is not a wait: nothing claims the primary recovered, the model alone
asks to return, and only the primary's own accepted reply counts as a return.

A submitted action is shown as pending until the task reports its application; a
saved Settings change and a still-pending task change are disclosed separately.
When a requested Settings save failed or is unconfirmed, resolving the wait
shows one toast with those facts; closing its card without resolution announces
nothing. Retries preserve the original request identity and payload. Revisioned
rows reject older observations, and resolved episodes never reopen. Terminal
authors retain exact live paid-review controls without reviving author status;
other waits end. A mailbox delivery failure keeps Retry request available for
that same accepted command, even while its application remains pending. The
current task attempt selects live actions; a previous attempt's retained pause
never makes a new working attempt appear to wait. Updates and history rebuilds
preserve the same keyed editor, its draft, persistence checkbox, focus and
reading position. Wait pickers omit Context controls on every render: their
action changes model/account only. Closing the chat disposes view resources
without claiming it stopped the task; continuation requires the Ouroboros
process to keep running.

A consciousness wake-up needs nothing of its own here: it is an ordinary direct
turn with its own task id, so its model wait, controls and temporary model
change behave exactly as an owner turn's and last until that turn ends; the
persistence checkbox still saves the consciousness role. `Started by: Consciousness`
names its origin in the block's details; its final bubble and tasks' cards
keep that origin, never a separate card, reused slot or vocabulary.

An already-delivered answer does not close a still-open post-task synthesis.
Consolidation or chooser waits use the same role controls in that task's
existing finalizing card. Pooled tasks, including API-only tasks, keep their
worker slot until post-work settles; the answer arrives early. Ordinary native
chat post-work holds no worker slot; its existing card keeps the same live
post-task model-wait controls after ordinary dialogue admission closes. Both claims follow the host's live owner
and post-task checkpoint, not the presence of answer text or a cost estimate.
Failed main work stays visibly failed after history reload while post-work
controls remain live; the unfinished checkpoint never erases the outcome.

### Widgets board

The Widgets page is a board of live cards. The owner arranges the board; each
widget's content belongs to its author. The board scrolls with crisp edges: no
fade dims a card where the list cuts it.

- **A masonry in the owner's order.** Cards are packed into columns like a
  Pinterest board: in the order the owner set with each card's move handle,
  every card goes to the lowest free place its width fits, so cards of different
  heights stack without a row layout's voids. The board takes its column count
  from the list's width and the widths its cards ask for; a board where the
  owner set no width is exactly the board before owner widths existed. A new
  widget joins the end of the order; the order is stored when the owner moves a
  card, so after a window reload the widgets added since the last move follow
  the stored ones alphabetically. A widget whose skill is off is not drawn; its
  place in the order (moving the other cards keeps its slot) and its width stay
  stored for its return, and nothing is reserved meanwhile.
- **Width is a column span, the owner's over the author's.** A card spans 1,
  2 or 3 columns, or Full width. Until the owner picks one, the author's `span`
  decides (`1` one column, `2` two), and the board may widen a lone one-column
  card beside wide ones to fill its lane. The board never widens or narrows a
  width the owner chose, except that no card is wider than the board: 3
  columns on a board of 2 is the whole row. Full width is always the whole
  row. Because the column count follows the widths the cards ask for, a
  card's width can make the other cards' columns wider or narrower. The
  owner's choice holds for that card on every screen of the installation
  until Reset size returns the author's default.
- **Widths that do not tile leave gaps; the board does not backfill.** A
  card that cannot fit beside the cards above it starts below them: a
  3-column card after a 2-column one on a 4-column board leaves the column
  beside the first card empty above it. The board never repacks cards out of
  the owner's order to hide such a gap; the owner closes it by changing a
  width or the order.
- **Height follows content.** A card is as tall as its widget: a table shows
  every row, a framed widget its declared or reported frame height
  (`docs/CREATING_SKILLS.md`). No card is pinned to a height. When a card
  grows or shrinks, the masonry packs again, so cards after it can move,
  even to another column.
- **A narrow list is one column.** When the list is too narrow for two
  columns of 280px — a phone, the Telegram mini app, a narrow window — the
  cards stack at full width in the same order and widths do not apply; the
  board returns when the list is wide again. The list's own width decides,
  not the viewport.
- **The card menu works on every surface.** Every card's ⋮ menu carries Size
  (1, 2, 3 columns, Full width and Reset size) beside a framed card's launch
  policy, so a touch screen, a keyboard or the mini app — no hover there, and
  a vertical drag collapses it — can size any card; on a narrow list the menu
  says that widths apply when the list is wide. Wherever a step would change a
  card's width, its right edge can also be dragged between the widths the
  steps give it (Escape cancels) or stepped with its arrow keys; the only card
  on a board, which every step leaves the whole row, offers no edge.
- **Arranging never restarts a widget.** Reordering, resizing or a change of
  the column count changes only where a card is painted; a running widget
  keeps its frame and its state. Keyboard focus keeps the card's original
  position until the window reloads.
- **Stop holds until Start.** A framed card the owner stopped rests as its
  facade with Start on this device across reloads, until Start or a launch
  policy of Auto or Keep running starts it. A card whose last start the page
  did not outlive rests the same way, with its one line (Skills in the page).
- **One arrangement per installation.** Order and widths are owner UI state
  stored on the server, the same for every window and device of the
  installation; when two windows arrange at once the last change wins. A
  width that failed to save stays on screen and says so under the list until
  a later width change saves it; nothing retries on its own.

## 6. Account group / row anatomy

For a repeated identity row (a connected agent account, a subagent row,
a server entry):

1. **Classification chip** — neutral pair, `--type-meta`. Only where the row's
   family is not already expressed by the group it sits in; inside a per-family
   card the chip repeats the header and is dropped.
2. **Name** — `--type-body` semibold, `--text-primary`. The one primary thing.
3. **Identity detail** (email, plan) — `--type-meta`, `--text-meta`.
4. **Status** — dot + a factual label from the status pairs. On Agents cards,
   use one short availability label; its sentence is the title, and is said
   under the head when it adds a specific pin, model or limit. Other row types
   keep the status wording their own contract requires.
5. **Meta line** — `--type-meta`, `--text-meta`, on its own line under the
   name. Quantities are stated in human words ("38% used · resets in 2h"), and
   an instant is humanized. A row never leads with a raw ISO timestamp.
6. **Actions** — docked right in a slot of their own, legible at rest. An action
   rendered at secondary ink reads as disabled: its label is `--text-primary` (a
   setting's name and an unchosen option take the secondary ink, bible principle 4).

**The main view carries understanding, state and action; justification is
disclosed where needed.** A row reads identity, state, facts; its primary choices,
exceptions and owner actions stay visible, and a value keeps its honest form
(`up to`, `Tracked:`, unknown). A rule every row shares is said once for the
section; history, provenance and why a state or value reads so wait in details
pointer, touch and keyboard open, never hover alone (Accounts: metrics under Diagnostics,
explicit store failures visible; exact program text under Details). New copy serves a decision or error, adding meaning no
control or status already carries: a checked box is not re-said beside it, and
an unsaved draft is said once by the editor or its host page, not on every row. A conditional
line never moves the control that caused it: controls own stable slots, and a
line an edit toggles opens beside or below them. Clarity comes from placement,
never from smaller text or a line quota; real exceptions all stay.

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

Context details show the server-built sourced window line: "set in Settings"
labels a manual value; the Clear confirmation button removes the route acknowledgement.
Account changes withdraw old metadata during pending or failed catalog reads.
Unknown stays unknown. Catalog updates preserve the edited field and caret
without assigning a model. `model_roles.js` and `model_roles.css` own the shared
Settings/wizard editor.

A source is chosen, never spelled. Every surface that assigns a model — the
Models roles, Available subagents (reviewers included), the first-run wizard and
the quota-wait picker — offers one grouped source select with the same groups
in the same order: Subscriptions · models, API keys (one entry per provider with a stored
credential, then one disabled pointer to Accounts; a saved choice without a
credential stays selectable as "(no key)"), Agents · sessions where a session
is possible. The model chooser lists only the chosen source's catalog, so a
suggestion's transport is the selected source; any id can still be typed. The
stored spellings (`provider::model`, `claudexor::source=model`,
`harness=model`) are serialization authored by the editor: never required from
the owner, never a field placeholder or help-text instruction, never the
primary displayed value; the exact stored id may appear in secondary details
(Agents use Details & history). A configured-subagent reference is the one place a stored spelling
names a thing: a roster row is labelled by its handle — its route target plus
the facets that row really runs with, defaults omitted — because a friendlier
stored label rots as soon as the owner re-points the row. The route identity chip names the source (API · OpenAI, Codex · model,
Claude Code · agent), not the channel alone. A last-run receipt is shown
against the route that produced it: when the row's route changed since, the
line says so and names the earlier route.

Subscription model sources and agent sessions never imply each other's model
inventory. Source ids are opaque; the model-sources catalog names the
credential harness. Saved subscription-model account pins survive catalog gaps
and unrelated saves; direct API-key models do not offer a subscription-account
pin. Catalog entries are suggestions, not account-specific entitlement or
context evidence. Changing a model or account never changes the delivery kind:
an API catalog row keeps its explicit Native or Packet choice regardless of its
subagent id, while a session row retrieves with its native tools. Session Access
uses the catalog's existing select: Full system access (default), Working files,
or Read only. Writable reviewers check private copies; their outputs remain
evidence without product integration. Access never changes delivery. Catalog refreshes
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

### Advanced settings

The panels a new owner opens first (Models, Providers, Agents, Behavior) show only what an owner touches in the first week. Anything a typical owner never needs, or would have to look up before using it safely, lives under Advanced: deployment plumbing (bind hosts, process pools, runtime limits), transport and trust knobs such as the extra CA bundle, MCP servers, and the rarely used provider cards, which stay on the Providers page but fold under "More providers". A control that needs a paragraph of explanation before it can be used safely is an Advanced control by definition. The engineering seam is the `advanced` settings panel and the `advanced: true` card flag in `web/modules/settings_ui.js` (DEVELOPMENT "Onboarding and Settings surfaces").

## 8. Migration state

The scale is applied to complete component families and declared page regions.
Using a migrated field inside a historical page does not claim the whole page
has migrated. Migrated today:

- `web/ui.css` (shared palette, fields, buttons, status/chip recipes and
  menu/chooser chrome, used by both top-level documents and optional author pages)
- `web/settings.css` (settings shell, model/effort cards, MCP cards)
- `web/onboarding.css` (the whole first-run wizard)
- `web/model_roles.css` (shared role editor)
- `web/room_work.css` (room work disclosure, rows and reading return)
- `web/cards.css` (living cards in the feed)
- `web/style.css` regions paired by `design-system:migrated-begin` /
  `design-system:migrated-end`: harness accounts; Dashboard → Updates (status
  card, one action row, collapsed Recovery with one restore list), Storage and
  Evolution;
  the Skills page (Installed, Browse, Sources, the detail sheet, its dialogs);
  chat typography, foreground and status colour. Chat geometry retains local
  literals per the viewport reserve contract; shared palette channels keep
  translucent glass coherent across themes.
- the global `.muted`, `.form-section h3` and shared `.ui-status` tone rules

The remaining page-specific typography in widgets and logs is
historical. Their adopted common controls follow the shared
family; unrelated page rules keep their literals until their own pass. Migrate
each selected family completely and remove its replaced recipes in the same
change. Do not introduce a half-tokenised second field family or describe a
control adoption as an all-page redesign. The semantic status/action/notification
contract already applies everywhere.

`tests/test_web_typography_static.py` guards the migrated set only. Extending
the guard and migrating the corresponding family or region are the same commit.
The bible's materials (objects, trays, sheets and the three capsule tones) reach
every page in both themes through the "Airy dopamine" blocks of ui.css, style.css and
the page sheets; that mapping is not a typography migration of those pages.

### Author freedom

An extension may use the optional shared buttons, fields and status functions
inside its own module or route-iframe page, override them, or design a completely
independent interface. `.ouro-ui` supplies font and native dark-control context;
the named classes opt controls into the recipes, with no page-wide reset.
The kit reads the installed source at a new mount; retained frames keep the styling
they loaded. A module may opt into the existing `OuroborosWidget.onTheme` signal
and apply its own `data-theme` rules, but the kit introduces no theme polling,
forced remount or mandatory visual conformance. Author layout, validation, operations and loading feedback remain
author-owned; the small source recipes are in `docs/examples/author_ui_kit/`.

---

## 9. Notifications

This section is the ONE canonical statement of when Ouroboros pulls the owner
back to the interface. `docs/DEVELOPMENT.md` points here and adds only
engineering rules; no second policy list may exist.

**Meaning.** A notification means *come back, there is something here for you* —
never *look, I am still working*. It exists so the owner can leave the window
and still be reached by a question or a finished task.

**Startup & background.** Behavior owns host lifecycle choices; Appearance is client-local. Sign-in startup is opt-in and names the host computer, also in remote clients. OS registration is truth, never a saved-settings guess: another copy, OS-disabled entry and unavailable build remain visible. Automatic startup preserves Panic and saved pauses; an explicit owner start releases Panic. *Keep Ouroboros running in the background when the window is closed* is also opt-in, stored on the host. First close asks once whether to keep tasks, schedules and Telegram running; its answer sets the checkbox, dismissal quits. macOS uses Keep running/Quit; Windows uses standard OK/Cancel and explains which keeps running. Closing with background on hides only behind a live indicator, so a hidden process always has a visible way back; otherwise it quits. Windows keeps one icon even with background/notifications off: after runtime and PID-lock/automatic-start checks, before Git/bootstrap. macOS shows its menu-bar item and Dock while background is on. The menu names starting, confirmed work, errors, unconfirmed activity or no answer; it offers Open, Panic and Quit. Early Panic stops installation work, including the retained daemon, and persists the stop marker; failed stops/writes are disclosed. Quit, Cmd+Q and sign-out quit unasked. Automatic start hides only with both checkboxes on and a live indicator; opening during boot shows the window. There is no separate start-minimized setting. Linux says background is unavailable and closes to quit. Android retains automatic boot by default with Panic preservation. This card is last on Behavior; browser/Docker hosts explain it without offering unsupported controls. Unavailable controls explain why in meta ink, without a warning dot; duplicate reasons appear once after both. Changeable states (another copy, OS-disabled entry) use warning; failed reads/writes use error. Notes align with checkbox text, status dots hang beneath the checkbox.

Panic sends immediately, without confirmation or an ordinary chat bubble. Acceptance clears only unchanged sent command text; unknown/refused delivery keeps it. Newer edits, reply and attachments stay. Desktop uses its independent stop bridge, also while the socket reconnects: no veil covers the app or its Panic. Browser uses its captured socket, never a queue/reconnect. Missing acceptance says “Panic was not confirmed”, pointing desktop to its tray/menu bar and browser to the host computer.

**Telegram completion messages are separate from these client alerts.** The
existing completion toggle Off suppresses every additional terminal line, failures
included; On sends endings, naming material problems before review provenance.
An otherwise finished task with only technical caveats says *finished*, without
a warning or success icon, and points to review and verification details in the
app. It never claims approval. Missing promised output, unresolved work, delivery
uncertainty and unmet or unknown review obligations remain material. This phone
projection never shortens full card reasons or rewrites outcomes. The status-only
Main mirror follows the toggle too, but preserved intermediate content does not.
Ordinary replies, files, questions and reply-editing Silent Mode remain independent. Unrecoverable old mirrors keep their text rather than guessing it is status-only. There is no new mode, cross-channel deduplication or retroactive rewrite.

**Telegram typing means a reply is being prepared for that chat.** It is a
native header cue, not a notification. While a reply the owner asked for is
actively prepared and would reach a Telegram chat under the current mirror
settings (Main and Project replies included), the bridge renews Telegram's
typing action there, once per chat however many tasks back it, every few
seconds at most. Ouroboros's own model and tool work counts; while that work
waits, what it waits on counts when it works: a helper's own work, a delegated
run the host currently observes running, or a review seat not waiting for model
access. A running run's provider retry inside its engine carries no fact, so it
reads as running. An intermediate message does not end the cue, and work
resuming after a wait brings it back with no new start. By itself a question
waiting for the owner, a wait for model access, a timer- or mail-only wait, a
run queued or paused on its own question, Pause or Stop now renews nothing; a
handed-off answer stops the helpers too, and background work nobody asked for
and unknown state never count. An action already sent when the answer leaves
may still be accepted after it, and Telegram may show the last action for up
to five seconds more; no message, card, percentage or Stop button is added.

**When the client runs.** Notifications are a property of a running client, and
there is no push channel. On the desktop the Telegram bridge runs inside the same
server, so it reaches the owner while the window is closed only in background
mode; quitting ends it together with the server.

**Focus does not suppress, and neither does a closed room.** While a category is
on, its event notifies whether or not the window has focus and whether or not
the relevant chat is open. A Project's question reaches the owner even when that
Project was never opened in this session — which is the whole point, and the
reason the subscription belongs to the client rather than to a room. That is
deliberate for the first version: we measure how it feels before adding clever
exceptions.

**Two authorities, one surface.** Every category is one of two kinds:

| Category | Kind | What decides |
|---|---|---|
| A question or decision is waiting | required | a confirmed lifecycle fact: the question carries a positive wait |
| A task finished or stopped | required | a positive typed terminal fact on a ROOT task |
| Messages Ouroboros sends while working | LLM-first | Ouroboros chose to speak outside the turn's answer (a proactive message in its room or as a Main notice, or an optional question) |
| A note Ouroboros left for this moment, or a granted skill's notice (same toggle) | LLM-first; owner grant | when it wrote the note, Ouroboros chose to leave these words for this time; a skill speaks only under the owner's `notify_owner` grant; the host only shows the signed `reminder`/`skill_notice` row |
| Ordinary replies in Main | separate toggle | an ordinary finished reply in the Main thread |

*Required* means the application asks for delivery from its own state rather
than relying on the model to remember, not that delivery overrides system
permission or platform limits. The separate soft fallback tone's Focus limitation
is stated below.

*LLM-first* costs no extra model call and needs no new host field: the signal is
that Ouroboros already chose to speak outside its answer
(`send_user_message`) or to ask something optional. The decision to send is the
mind's; the client only carries it. Stated precisely because the tool contract
asks for such a message at the START of long work as well: this category means
"Ouroboros said something while working", not a claim that it weighed whether
to interrupt you. Judge it by use and turn it off if it is too chatty.

**What never notifies.** Progress, a transient failure, one tool's error and a
reviewer's finding are not notifications; they stay in the transcript. A child
task never notifies the owner: it escalates to its parent, and only the parent's
own message or the root's terminal can ring, and a run that finished quietly
(`quiet_effective`) does not, unless it publishes a host notice. Machine partitions — the hidden
chat and agent-to-agent traffic — never reach a banner. A Project's room
notifies once this client knows it; any other ordinary conversation, including
one arriving over an external transport, is treated as the owner's exactly as
the Main thread itself treats it.

**A terminal is not always an ending.** An update or restart teardown reports a
task as interrupted and then requeues it; that is not a finished task and does
not notify, so the task's real completion still can.

**One ending, one notification.** An ordinary reply and the terminal of the same
task are one event, not two: whichever arrives first rings, and the other is
collapsed. The same holds for the several wire shapes a finished task has.

**System notifications first.** Where the desktop app offers them, an alert is
handed to the operating system's own notifications: sent even while the window
is in front, titled with the kind of event, sounding with the system's own sound,
and kept where that system keeps notifications. The system taking it is not a
promise that anyone saw a banner: its notification settings, Focus and Do Not
Disturb still decide. macOS uses `UNUserNotificationCenter`, whose one permission
question is asked only from the owner's gesture — switching notifications on, or
Test — never by an alert, which falls back while the question is unasked; Windows
shows a balloon on Ouroboros's one notification-area icon, presented as a system
notification on Windows 10/11; a source handoff can change the window/taskbar
identity to Python. Handoff removes the old icon before its successor starts;
a gap or Explorer ghost after a crash remains possible; Linux uses the freedesktop notification service when
the desktop runs one. Not yet allowed, denied, unavailable or failed is a typed
answer, never silence (an error the system returns instead of asking is that
error, not a denial): that alert falls back to the browser banner or the in-app
toast. One the system took without answering in time may still appear, so
nothing else is shown; its click opens the newest accepted source on that icon. In-app
updates do not replace the desktop app, so an app built before this capability
falls back and Settings says that system notifications need the current app.
Android's host keeps its own channel for its own service; its page has no system
notifications and says so.

**One sound.** A system or browser notification owns its sound and volume; the
page adds none beside it. Before native dispatch the page refreshes system
permission, then rechecks its own current choices. A submitted notification does
not prove permission: macOS may accept a request it will suppress. Unknown
delivery may still appear and gets no fallback.
Only a definite failed hand-off, followed by a fresh positive system permission
and sound-setting read, may use the soft app tone when no banner or other sound
was shown. Explicit denial, unasked or unknown permission stays quiet; a cached
grant cannot overturn a refusal. The app tone has its own volume and **may not
follow Focus / Do Not Disturb**. Neither the old Glass sound nor the launcher's
standalone beep is a fallback.

Sound off requests silent delivery, including Windows' `NIIF_NOSOUND` balloon.
Every page-controlled dispatch rechecks current notifications, category and Sound
choices after any wait; a silent decision never becomes audible, and a pending
tone cannot become louder. Already handed-off effects cannot be recalled.
A legacy launcher with only `request_attention` keeps its silent window cue.
One with `notify_owner` but no system-notification method is never asked: its
hidden-window balloon may sound despite mute. It keeps the browser banner or
toast, without the launcher's balloon, Dock mark or window raise.

**Each open window is its own client.** Settings, permission and the
duplicate-collapsing that keeps one event to one notification all belong to one
running page. Two windows of the same device — the desktop shell and a browser
tab — are two clients and can each notify for the same event. That is the
honest consequence of per-client settings, not a bug we have hidden.

**Click goes to the source.** A notification opens the question or the result it
is about — the Project room and the exact question when it has one, otherwise
the conversation. Project identity comes from the full registry, including rooms
outside the sidebar. An unavailable destination leaves the view in place and says
so. A Main-transport notice opens Main, including Telegram; numeric addresses
never imply a Project. A system notification first opens the window, hidden or not,
then the same source. No reply is composed from the banner. Windows names no balloon
in a click (owner decision: one icon), so any balloon click, an older entry in
Windows' list included, opens the newest balloon's source; when that is the
fallback banner the indicator shows for a hidden window (no system
notification shown), the click opens the window as it was left, no navigation.

**Content is private by default.** Only the kind of event is shown until the
owner turns message text on, because a banner can appear on a shared screen.

**Deliberately absent.** No numeric badge, repeated reminder, inline reply,
Telegram escalation or attention after quitting. A current desktop launcher may
raise a visible window as fallback, without its own sound. A deliberately hidden
window is never raised by an alert: Windows may show a silent indicator banner,
macOS marks and bounces the Dock icon; the indicator or Dock opens the window.
With browser notifications allowed, the page asks this bridge first because a
browser click cannot unhide the window. A visible window keeps the browser banner.
Attention cues prove no system delivery; unsupported and legacy launchers fall
back as described above.

**Settings.** The controls live on **Settings → Appearance**, under the theme
block, and are stored per client exactly like the appearance choice: the desktop
window and each browser keep their own, nothing reaches the server. Under Sound,
**App tone volume** controls only the app tone, never system volume. Automatic
fallback requires a client that can report positive system permission and sound
settings (currently macOS); elsewhere the control affects Preview only. Moving
it is silent; **Preview tone** explicitly plays the chosen tone, even without
notification permission, but not with Sound or notifications off. Both use the
existing controls and button styles, not another panel. A test
button is the honest way to see what this system actually does with a
notification, including a denied permission, and in the desktop app it is the
gesture that lets the system ask. The status line names the surface this client
is using — the system's notifications, a browser banner or the in-app toast —
keeps the system's permission apart from what became of the last alert (taken,
unconfirmed or fallen back), and, in an older desktop app, says that system
notifications need the current app.

**Known limits of this version**, stated rather than discovered later:

- A task retried under the same id has one conclusion as far as notifications
  are concerned; a second terminal for that id stays quiet.
- Duplicate collapsing is bounded (the oldest keys are forgotten after a very
  large number of events in one session), so a frame for a long-past event
  could ring again.
- A child task is recognised from the delegation facts its traffic carries (its
  declared lineage, or the executor enrichment on its own terminal). Nothing on
  the wire declares a task to BE a root, so a task with no such fact is treated
  as one: requiring proof of root-ness would silence every finished task. The
  narrow residual is a child that carries no delegation fact at all and whose
  first observed frame is its terminal — it would notify once.
- An event that happens while the socket is down never rings: reconnect replays
  history, and history is deliberately silent.
- On Windows a click opens the newest balloon's source whichever entry was
  clicked, and a balloon stays in Windows' list at most until the app quits;
  nothing is shown again. An exit removes the icon within a bounded wait: a
  shell that does not confirm in time can leave it until the pointer passes,
  and an updated launcher's own icon can briefly appear beside it.
- An alert the system took without confirming in time can stay unseen: it is
  not repeated in the app, and Settings says so. On Linux its click cannot open
  the source, because the server's id for it never came back.
- A system notification clicked after its page reloaded only opens the window:
  click targets live as long as the page that sent them, with no stored ledger.
- What a system notification looks like is the system's choice: macOS's style
  for Ouroboros (banners, alerts or none), Focus and Do Not Disturb, Windows'
  notification settings, and a Linux server's optional actions (without them a
  click cannot open the source) and sound (without it the alert is silent);
  Settings names what that server lacks.

## 10. Shell

One interface serves the desktop window, browsers, phones and the Telegram Mini
App. The phone layout is the same elements arranged by one rule, never a second
set of screens, so a page added for the desktop works on a phone without phone
code of its own.

**One layout switch.** The shell has two layouts: *docked*, when the window is at
least 720 px wide, with the navigation column beside the content, and *stack*,
narrower, where the column becomes a drawer. `html[data-layout]` carries the
choice, set before first paint from the one constant `web/theme.js` exposes and
`web/modules/layout.js` reads; shell rules key off the attribute, components adapt
to their own box with container queries, and no page adds a width breakpoint of
its own. The surface fact sent with a message reads the same switch. On docked,
the Project panel lies beside Main at its own width over the dimmed room; its
handle resizes it apart from the sidebar's, and it is never wider than the
content area.

**Resizing.** A handle drag writes only the dragged box's width, once per
displayed frame; the sidebar's column, the dimmed room and the panel's edge
follow that box by layout, and the width is stored once, when the drag ends.
While a feed's width keeps moving (a drag, a live window resize), the rows far
from the screen keep the size they were last laid out at and only the rows near
it are laid out again; once the width holds, every row is laid out at the new
width and the passage being read stays where it was. A single change of width (a
window jump, a layout switch) is laid out at once.

**Navigation.** Places come from one registry (`web/modules/pages.js`): it renders
the navigation rows, names the titles and feeds the shell gate, so a new page is
a registry row and its module, checked at phone widths with no further work. The column
has a pinned head (Main Chat), one collection (Projects, the column's only
scroller) and the tools right after it (Files, Skills, Widgets, Dashboard,
Settings, then the version and the update), which reach the bottom edge only
when the Projects do not fit: free room stays under the last row, never between
them. Main's row carries its room's status mark at the Project rows' x. A Project's
menu opens from its ⋯ (a fine pointer), a right-click, Shift+F10 or a long press; it
first says the row's state as its mark and words, then offers Pin or Unpin; pinning
neither opens nor reads it. Pinned Projects lead the one list, and one `--space-2` of air before the next
shown row is their only boundary: no caption, glyph, tint, rule or column, so
nothing reads as one more Project; the row's tooltip and accessible name say
Pinned. Within each group unread leads, then recency; the column's places follow §5.
After Unpin, a room outside the recent sidebar slice disappears from that slice,
not from the registry. The Projects header collapses the whole list or opens it:
collapsed, only the pinned Projects stay, in their order and with their air; every
other row hides whatever it is doing, the open one included, and the header keeps
its mark and unread count for them. The choice is remembered on the device; a row
that hides while it or its menu holds focus closes its menu and hands focus to the
header. No label is set in capitals. A short window first compacts rows (fine pointer only), then lets the
whole column scroll with Main pinned; a tool never becomes an unlabelled icon. One
place is chosen at a time, and pressing a place opens it, never closes it.

**The drawer.** In the stack layout the room is a card the drawer pushes aside.
The same column lies under the room; opening slides the room right by three
quarters of the screen (never more than 360px), as a card with the screen's
rounded corners, a soft shadow and its words faded, and brings the column up from
under it with a little parallax. Nothing darkens. The quarter of the room left
showing is the way back: a tap or a sideways finger on it closes the drawer. A
Project open over Main is the room then, and moves the same way; on the desk the
column stays docked and nothing changes (owner 19.10). In a room (Main, an open Project) ☰
is the first control of the composer's row, beside Swarm and under the thumb, and
steps aside with the others while an open capsule has the line alone; on a page
without a composer it sits in the leading slot of the header. While the drawer is
closed ☰ carries one 7 px dot for Main and the Projects, the room on screen
included: amber while one of them waits for the owner, else crimson while one has
unread messages; its accessible name says which. The
drawer's × sits in its head, where a page's ☰ is. The room's quarter,
×, Escape, Back and choosing a place close it. A drag only speeds this up: a
sideways finger anywhere on the room, from the screen's very edge and over its
buttons, links, header and composer, pulls the drawer with the finger, and holds
it while the row under the finger is redrawn, a browser bar changes the window's
height or the thumb drifts up or down (once the drag is sideways, the page cannot
scroll under it); only a field, a slider, media, selected text and anything that
scrolls sideways keep their own finger. Dragging it back closes it
(`web/modules/drawer_drag.js`, switched off by one constant); the room card and its
veil follow the same finger. A menu or dialog that opens under the finger, a long
press's row menu included, ends the drag before it starts. Every gesture has a
visible control, except a Project row's menu on touch: phone rows have no room
for ⋯, so there a long press opens it, and the ⋯ stays in the row for the keyboard
and screen readers, appearing when focused (`web/modules/row_menu_gesture.js`;
a feed row keeps its own press, which sinks and lifts it).

Where the system or the host owns the very edge (Android's back gesture, the
leftmost ~30 pt in Telegram on iOS), its gesture wins; a start a little further in
is the drawer's.

**The drawer's motion.** Under the finger the room card moves 1:1 while the
column comes up from under it and the card's words fade; nothing darkens. Pulled
past fully open the card gives like a rubber band, the column's edge filling the
gap. Released, or when the system takes the touch, it settles the rest of the way
at the finger's speed (180–400 ms); ☰ opens it in 340 ms and × or the room's
quarter close it in 280 ms, on the iOS sheet curve (`--ease-sheet`). A thumb that
rests before it lifts is no fling. The quarter's clear scrim comes and goes with
it. Reduced motion keeps the finger tracking and drops the rest.

**Moving between places.** On a phone a place picked in the open drawer selects
its row at once (no travelling pill or hold) and changes inside the room card as
the drawer closes: the leaving room's words fade out and the arriving room fades
in with one breath (scale .97 → 1) while the drawer's own close brings the card
home (owner 20.10, 4A; 21.10; `web/modules/shell_motion.js`). The stack,
history and Back change at the tap; only the covered room's inert waits for the
card, because it restyles the whole room. A Project room that is left keeps its
rows until its exit has run and is the same room when reopened meanwhile. Docked,
navigation stays instant; the Project panel comes in from the window's right edge
once its room has painted (or after a read's 300 ms bound, with its loading row)
and goes back there whole, only its scrim fading (owner 20.10, Q2 A). A dismissed
toast fades where it stood while the ones that stay glide into its place. Moving
status marks share one document phase while keeping their three-dot delay
(`web/modules/status_mark.js`); repainting an unchanged mark does not restart it,
and under the closed phone drawer they rest. Reduced motion keeps only short fades
(the arriving room, the docked panel) and stops the wave. No veil covers the app
while it reconnects: the room's bar says Reconnecting….

**Haptics.** Inside Telegram, and only there (its HapticFeedback; a browser and the
desktop app have none), a few moments are felt, one meaning each: the drawer,
released, latches open or closed (a light impact); a level steps or Swarm flips (a
selection tick); a long press opens a row's menu, a swipe on a message reaches its
reply point (again after pulling back under it) or a widget is picked up to be moved
(a medium impact); Stop on a work is pressed (a rigid impact, once); the owner's own
action does not go through (its error toast: the error notification; a notice he did
not cause stays silent). Sending, copying and other toggles stay silent, as in
Telegram, and nothing depends on a haptic (`web/modules/haptics.js`).

**Layers and dismissal.** The drawer and the Project panel are layers in one
stack (`web/modules/layers.js`); dialogs, menus and the Shade keep their own
Escape and outside-press dismissal. Escape, ×, a press on the scrim close the
topmost layer and only it; the system or browser Back and Telegram's Back close
what is on top: a dialog over a layer first, then the layer. While a layer is
open the page holds exactly one same-address history entry, so Back closes it
instead of leaving the app; navigation itself is still never written into the
address. A modal layer makes the rest of the shell inert (never what opens over
it) and returns focus to its opener on close. The shell draws one close glyph and
one More glyph (`web/modules/page_icons.js`).

**Page header.** One component (`web/modules/page_header.js`) draws every page's
header and the Project panel's, as one row except the narrow toolbar below. No decorative icon.
On docked, a page the column already names shows no title: its tabs or Files'
location take that place and its actions the end, on Main's line, so the bar has
two ends as Main's does; tabs or a location that do not fit scroll inside it (a
location rests at the folder you are in, no folder wider than the bar's start),
and in the macOS window its empty parts still move the window. A header with nothing to show draws
no bar (Widgets keeps the plain band). A Project shows its name. In the stack
layout the row is at most 56 px: ☰, the place's name, its status, then Panic and
⋯, with the page's tabs, or Files' location and its actions, under it
(`bindPageHeader` moves those nodes); the name wins over the status word, which a short row
drops whole, keeping its mark. Navigation sharing a toolbar also returns below the bar when the
content column is too narrow, including while the sidebar is resized; the toolbar wraps in its
own row. Files' location and actions return above the list, folder names fit and actions wrap. Whenever the bar itself is narrow, on a phone or in a narrow window,
Restart and the budget move into ⋯ — the same controls, placed by the bar's
width. While a mode it holds is on (Evolve, Consciousness), ⋯ takes its tone,
with no corner dot. In Main and Project, the bar has no continuous backing strip:
the title and status share one floating capsule (`--chip-float`, non-shrinking
mark and caret with inline inset), and each command is its own floating capsule
(owner 21.10). A failure or approximation note hangs under the bar; a page
has no caption line, its sidebar row and its tabs name it.

**Doors and containers.** An object whose only job is to take you somewhere is a
door: the whole object is the target and one state layer covers all of it (a file
card, a Files row, a navigation row, a Project reference). An object that carries
reading content or several actions is a container: its body is text and never a
target, and only its own controls react — a message, a question, a Working card
whose "Details" is its one disclosure, a work card bound to a Project whose
footer strip is its door. Hover never tints part of an object, and hover paint
applies only where a pointer can hover (`@media (hover: hover)`).

**Touch and containment.** Under a coarse pointer every target is at least 44 px
(`--touch-target-min`); a control whose look is smaller keeps its look and gains
the invisible hit area of the one `.ui-hit-area` recipe in `web/ui.css`, which a
link inside a sentence takes too so the sentence keeps its lines. Nothing that
acts is hidden until hover there (a Project row's menu is reached by its long press).
No page scrolls sideways at any width; only code, tables, tab strips and a bar's
Files location scroll sideways, inside themselves, with a fade where content
hides. Text wraps before it widens a column.

**Telegram.** Inside the Telegram Mini App the page tells Telegram it is ready,
fills the screen, turns off Telegram's swipe-down so lists scroll, shows
Telegram's Back while a layer is open and its Close otherwise, gives Telegram's
header and bottom bar the room's colour, follows Telegram's light or dark as the
System appearance, and respects Telegram's safe areas
(`web/modules/host_telegram.js`). Everything else is the same page.

Enforcement: `tests/test_design_system_gate_browser.py` visits every registered
page and every Settings tab in WebKit and Chromium at 360, 390, a Telegram-like
390 × 780, 719, 720, 721, 1280 and short windows (no page scrolls sideways, coarse
targets reach 44 px, default-width bars stay one row, one scroller per sidebar, Projects
scroll by touch); `tests/test_shell_layout_static.py`,
`tests/test_shell_header_static.py`, `tests/test_shell_drawer_static.py`,
`tests/test_shell_tokens_static.py`, `tests/test_pages_registry_static.py` and
`tests/test_design_system_css_static.py` keep the one switch, the registry, the
glyph set, the breakpoint ratchet and sentence case; `web/tests/layers.test.js`,
`web/tests/drawer_drag.test.js` and `web/tests/host_telegram.test.js` pin the
behaviour.

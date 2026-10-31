# The craft floor

Checkable statements. Each is a prompt to look again, not a gate: a measured pass proves nothing about hierarchy, and a deliberate exception named in your reply is allowed. The product's design document outranks every line here.

## Hierarchy and layout

- One primary thing per unit (screen, card, row). Its rank comes from size, weight, colour or position, not from dimming its neighbours.
- Left edges of text and controls in one region share two or three lines, not six. Count them on the render.
- Spacing comes from one scale (a 4 px base is common); gaps between groups are clearly larger than gaps inside them. List computed values that fall off the scale.
- Grouping works by proximity first. Add a container only when proximity is not enough; fills do not stack, and nested boxes signal weak grouping.
- Repeated elements are identical: same edges, padding, radius, baseline; the recurring control sits in the same place.
- Nothing widens its column. Horizontal scrolling lives only in the wrapper that owns wide content (a table, code, a tab strip).
- A control never stands alone in a row because a wrap left it there; a control never widens for a long label without a plan.

## Type

- One family, or two clearly different ones; few sizes (app screens use roughly four to seven); weight and size carry hierarchy.
- Body measure around 45 to 75 characters; headings balanced across lines; numbers in columns use tabular figures and right alignment.
- No effective text under 11 px; inputs on phones at 16 px or larger so the page is not zoomed.
- Capital letters and tracked-out labels only where the product already uses them.

## Colour and depth

- Colour on objects says who, what state, what matters; the room behind content stays plain. Neutrals are tinted toward the accent so greys belong to the palette.
- One accent for the primary action and the chosen place; semantic colours for state; nothing else coloured for decoration. Two hues that differ only slightly read as a mistake.
- Text contrast 4.5:1 (3:1 for large text); non-text UI, borders of controls and focus rings 3:1. Over alpha, gradient or blurred backgrounds the ratio is unknown unless sampled from pixels; say so.
- One shadow scale by height (resting, floating). A state changes colour, never height; nothing moves a neighbour when it changes state.
- Both themes get the same care: check that every colour comes from a token and that the second theme is designed, not inverted.

## Motion

- Motion answers an action and shows what changed. At most one orchestrated moment plays by itself; nothing animates to look alive.
- Respect the reduced-motion preference; animate transforms and opacity, not layout.

## Interaction

- Hit targets at least 24 × 24 px with a pointer, 44 × 44 on touch; expand the hit area when the drawn shape is smaller.
- Keyboard focus is visible on every control; tab order follows visual order.
- Hover reveals nothing essential: touch surfaces have no hover.
- A control says what happens in a verb and keeps its name through the flow.

## States to see before you call it done

Empty, loading, error, the longest real text, many items, the narrowest width it ships at, both themes, keyboard focus, hover and pressed. Clipped text, a sideways page scroll, a bar covering a control and a floating element over the product are bugs, not styling.

## Signs of an unconsidered surface

Prompts to look again, never bans: identical cards for things that are not alike; a border on everything; a gradient, glow or texture behind content; everything centred; a tracked-out label above every heading; numbering where there is no sequence; emoji or random glyphs as the icon system; three near-identical hues; a control that looks like its sibling but behaves differently; captions, counters and disclaimers accreting with every revision; motion on every card; a serif-and-cream or black-and-neon look that appears whatever the subject.

## Words on screen

Words are part of the design. Call things what the viewer controls, in the viewer's own vocabulary rather than the implementation's. A control states its effect as a verb and keeps that name through the whole flow. An empty screen says what to do next. An error names the fault and the way out, without apologising. One element, one job: a label only labels, an example only demonstrates. Cut filler, template sections and remarks addressed to the requester; those belong in your reply.

## Greenfield notes

When no design system exists, the existing screens are the system: read their type, spacing, colour and depth from the stylesheet before inventing. If there are no screens, decide from the subject and the viewer's job, state the choices before writing code, and watch the clusters that generated pages fall into whatever the subject: a cream page with a high-contrast serif and a warm accent; a near-black page with one acid accent; a broadsheet of hairlines and dense columns; a kit of identical rounded cards with the same soft shadow; a purple-to-blue hero. Any of these is fine when the brief asks for it. Where the brief leaves an axis free, spend the freedom on the subject, not on a default. Quiet precision beats a visible risk that nobody asked for.

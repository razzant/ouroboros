---
name: ui_design
version: 2.0.0
type: instruction
os: any
permissions: []
description: "Design, change or show anything people look at (app UI, widget, mockup, HTML page or report, chart, icon): it fits the existing system, uses real content and states, and is judged by eye at the viewer's real size."
when_to_use: "Before designing or changing how something looks, judging screenshots of such work, putting visual variants before a person, or briefing another model that will draw. Not software architecture."
model_experience:
  what_model_sees: "A few lines of metadata while enabled. On read: a one-screen playbook; references hold the craft floor, the render-and-measure recipe, the author brief, a one-file comparison page and two browser-side snippets (measure.js, freeze.js). No tools."
  token_effect: "A few context lines while enabled. The body costs about two and a half thousand tokens when read; each reference costs tokens only when opened."
---

# Designing something people will look at

A surface looks designed when every choice on it is explained by what the viewer does there. Aim for a result that is good with nobody's feedback: a person's reaction is evidence about taste, never the step that makes the work good. This is a pattern, not a law: the request, the product's design document and the owner's words outrank it.

## 1. The job first

- Write the viewer sentence first: "On this screen <viewer> sees <what> and does <what>." It titles every page and brief you make, and every look below answers it.
- Name the viewer, their measured window (size, theme, engine, saved panel widths; the runtime context's `owner_client` fact names the owner's surface) and the one thing they do here.
- Authority, in order: the human's words; the product's design document and tokens (for Ouroboros's own interface `docs/DESIGN.md` and `web/ui.css`); the screens beside yours; this playbook; your habits.
- Census before invention: find every existing rendering of the same intent; reuse it or replace all of them. One intent, one control.
- Real content at its worst real length: the longest title, the long list, the empty list, the error. Real data from history exports and task results is allowed and preferred; only side effects on the live API are off limits. Invented data is at least as awkward, and you say so.
- A step from today unless a redesign is asked: change structure and behaviour, keep the materials people already approved. Recolouring the same boxes is not a design.

## 2. Composition essentials

- One thing first: the primary object or action is found without reading; the rest is quieter by size, weight or position.
- Shared edges: related things start on one left edge and sit on one baseline; correct by eye after looking.
- Spacing rhythm: one small scale, tight inside a group, generous between groups. Air is space, never faded text.
- Proximity before boxes: a border, fill or shadow marks a separate object; if everything has one, nothing does.
- Quiet surface: caveats, counters and test counts go in your reply, never on the surface.
- A plan that would look the same for another subject is a default, not a choice. The checkable floor, the tells and the words: `references/craft.md`.

## 3. Look, then let another eye look

- Open the delivered thing as its viewer will: the entry file or page, alone, at their size, theme and engine. Look at the first screen at 1:1, then at 2× crops of what changed; never a stitched sheet or a scaled grid.
- Then measure what eyes miss: overflow, clipped or covered text, bars over content, stuck controls, small targets, contrast (`references/browser.md`, `references/measure.js`). Measure after a paint; WebKit can report one width before the first paint and another after.
- An eye that did not draw it often sees what yours no longer can: when you judge it worth it, ask one to say what it sees first (what the thing is, who speaks to whom, what a viewer would do next). A reviewer given the frames, a critic you brief or a cold reader all serve; show the scene before your intent (the comparison page's clean view hides your names, notes and recommendation). When its words differ from your viewer sentence, the design is unclear: fix it before geometry.
- Findings are evidence, never a verdict. A green test, a saved screenshot or a check count is not a look. If you could not render or see the result, say so and give what you measured.

## 4. Putting a choice before a person

- One self-contained page made from `assets/compare.html`: one scene at a time at the viewer's real size, Now and two to four genuinely different directions swapped in place (keys 1–9, arrows; S side by side; C clean view), real product code and the real long data. A blink between pictures hides the difference being judged.
- One plain line per variant: what changes, what it costs. A is the recommendation, chosen for fit over novelty; "none of these" is always offered. A refinement shows Now and one proposal. A variant that moves away from something already approved says so.
- Demo chrome stays outside the product frame; nothing floats over the app. Open the file alone at the viewer's size and engine, switch every variant and look before sending.
- The message: one line on what is decided, the file plus one inline first-screen image for phone readers, at most three plain numbered questions, A first with its cost. No file names, test counts or process story.

## 5. After a choice

- A quick yes to a picture is a direction, not acceptance of what you build. Build from the chosen variant's code and data, then put the chosen picture and the built screen side by side at the same scene and size yourself. Fix what differs without a reason, send both with one line on what still differs and why, and go on: the merge does not wait for another yes.
- Record the reaction (§7) and close the loop item by item: what changed, what deliberately did not and why, what happens next.

## 6. Delegating the drawing

A delegated session may not see this skill: give it this file's absolute path and say reading it and its references is allowed. The brief template is `references/brief.md`; its first paragraph is the viewer sentence, coverage comes after. One direction per parallel author, or the variants converge.

## 7. The owner's taste

- Craft is general; taste is personal and changes. Before designing for a known owner, `knowledge_list(scope="global")`, then `knowledge_read(topic, scope="global")` for the topic holding their design reactions, if any.
- After a reaction, `knowledge_write(scope="global")`: the date, what was shown, their words verbatim, where it applies and where not, told, observed or inferred. Revise in place; a correction is evidence about that moment, not a standing rule until they make it one.
- Delegates return observations; only the mind writes memory. A taste note never becomes product law or part of this skill.

## Done means

A delivered page opens alone and fetches nothing; zero console errors; the awkward content fits; you looked at it at the viewer's size, in both themes when the product has two, and what you saw (and what any eye you asked saw) matches your viewer sentence; you can name what you saw and what you measured.

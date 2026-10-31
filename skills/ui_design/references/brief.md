# Briefing an author who will draw

A brief that lists coverage produces an archive. Write it in this order and keep it short; a delegated session has none of your memory and may not see installed skills unless you name their paths.

```text
VIEWER
On this page the viewer compares <X> and decides <Y>.
They open it on <surface> at <W×H> CSS px, <theme>, <engine>, saved widths <…>.
In the first ten seconds they should see <the real screen> and be able to switch Now / A / B.

THE OWNER'S WORDS (verbatim, dated)
"<…>"

WHAT THE OWNER HAS SAID ABOUT LOOKS (dated excerpts from the taste note, with scope)
<date>: "<words>" — applies to <where>, not to <where>.

THE SYSTEM
Design document: <absolute path>  (for Ouroboros's own interface: docs/DESIGN.md, tokens in web/ui.css)
Screens beside this one: <absolute paths or a running page>
A step from today unless this brief says "redesign".

DATA
Use the real thing at its worst real length: <absolute paths to history exports, task results, fixtures>.
Real data is allowed and preferred; the only limit is side effects on the live API.

DELIVERABLE
One self-contained HTML file built from <absolute path>/skills/ui_design/assets/compare.html,
frames inline ("</" written as "<\/" inside its JSON), opened alone at <W×H> in <engine>. Now plus <n> variants, one scene per decision,
A recommended with its cost in one line, at most three questions on the page, "none of these" included.
Not a README, a server, URL parameters, a manifest, a test count or contact sheets as the thing that is opened.

READ FIRST (host-provided read-only references; reading them and the files they name is allowed)
<absolute path>/skills/ui_design/SKILL.md

BEFORE YOU FINISH
Copy the delivered file alone into an empty folder, open it at <W×H> in <engine>, switch every variant,
look at the first screen yourself, then measure (ui_design/references/browser.md).
In your report: what you saw, in two sentences, with the engine and viewport; the measured findings you kept;
anything you could not look at.

RETURN
The file's path; the one-line note per variant; any observation about what the owner seems to like or
dislike, as an observation for the parent to record. Do not write the host's memory or edit its skills.
```

Notes for the parent:

- The viewer paragraph comes first and is the acceptance criterion; everything after it is material.
- Give one direction per parallel author; otherwise the variants converge. Tell each author which direction is theirs.
- Give the critic the same file the owner will get, in its clean view (key C) first: it says what it sees before it reads your
  names, notes and the viewer sentence.
- Keep the brief under a screen where you can; a nine-thousand-character brief about coverage is how an archive gets built.

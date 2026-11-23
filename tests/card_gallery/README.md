# Card gallery (test stand)

One loopback page where a developer picks a real Ouroboros card and one of its recorded states — an error, a long
wait, an empty answer, a long body — and sees what the owner would see. Every card is drawn by the product's own card
renderer at its real entry point, or by the product's own page composer (`createChatInstance`, `initWidgets`,
`initSkills`, `initConsciousnessDashboard`, the registered `agents` page). What the stand composes itself is only the
chrome around the card — the picker bar, the stage, a few host containers — plus the small wiring the app does not
export (the setup wizard's provider-panel calls, the Share registration `app.js` performs, the badge `chat.js`
`setStatus` already writes); the table's `How` column names exactly how each family is drawn. The product has no route
for it: only `tests/card_gallery_stand.py` serves it, on `127.0.0.1`.

## Open it

```bash
# Python with Playwright; the stand itself needs only the standard library.
python -m tests.card_gallery_stand serve            # prints http://127.0.0.1:<port>/
```

Pick a card and a state in the bar, or any scene under `All scenes`; `Source` names how the scene is drawn — its entry
point (or composed host) and the corpus it reads. `Next frame` delivers a recorded frame through the entry point the
live socket uses (a `quiz_state`, a `card_state`, a task's `chat` frame, the next work read). Links:
`/?scene=<family>.<state>&theme=dark|light`, `&frames=1` for the first frame only, `/?bare=1` for the same document
with nothing mounted.

## What the page reports (the tests drive this)

`window.gallery`: `ready` turns true only once the scene shows the state its address asks for — every read it started
answered, every wait it names met; a scene the page cannot draw keeps `ready` false and says why in `failed`, beside a
visible `.gallery-fault` row (a read the page says failed, a wait that ran out, an address naming no scene). `scene`
is the scene id the address names — the page draws that scene, or says it cannot, never another. `?scene=` naming one
the stand does not have is said on the page, with no card chosen; `?scene=` omitted draws the stand's first scene;
`theme` omitted takes the browser's colour scheme; `frames` omitted every frame. `scenes()` lists the stand's own
scene ids, and `bare` marks the bare document. `tests.card_gallery_stand.open_scene` waits for either terminal and
raises on `failed`, so a fault or a not-ready page cannot be captured as a good frame.

API calls name their scene per request: the page sets `X-Gallery-Scene` and `X-Gallery-Step` headers on its own calls
— no shared cookies, so two tabs never share a scene. `GalleryServer.requests` records each API request as
`{"method", "path", "scene", "step", "status"}`: the header values — `scene` verbatim (empty when absent) and `step` the
parsed integer (0 when absent or not one). Every other read is a 404 and every write a 405: nothing but a read is
answered. The sign-in card's start is answered from its recorded job inside the page, through the controller's own
transport, so nothing is posted; the waiting card then polls the stand for that recorded job (an ordinary GET), and
the failed card does not poll.

All corpora are read once when the server starts: the corpus endpoint and the API answers both serve that one snapshot
for the server's lifetime. An edited `--card-corpus` draft shows up on the next `serve`/restart, never inside a
running stand.

## Look at it (frames and measurements)

```bash
export PLAYWRIGHT_BROWSERS_PATH=<an existing Playwright browser cache>   # never an installed browser app
python -m tests.card_gallery_stand look --out /tmp/card-gallery [--scenes quiz.long,work.incomplete_read] \
    [--engines chromium,webkit] [--themes dark,light] [--viewports 1440x900,390x844]
```

Per scene, engine, theme and viewport: a first-screen PNG at device scale 2, the `ui_design` skill's
`references/measure.js` findings (read from the skill, not copied) taken at that initial screen, and only then a crop
of the card — the crop scrolls, the measurement does not. `look` validates every asked scene id against the stand's own
list before it takes any labelled image; a scene that faults, shows something else than asked or measures a different
viewport ends the run instead of labelling a frame — an address naming no scene is said on the page, never drawn as
another. Each frame in `report.json` records the scene the page reports (`shown`), the measured viewport and the
findings, beside the asked scene, engine, theme, viewport and any page errors. Playwright's own headless Chromium or
WebKit only: the stand refuses an executable outside `PLAYWRIGHT_BROWSERS_PATH`. These are browser frames, not the
macOS app window.

The `ui_look` tool (`skills/ui_look`) can look at the stand too, through its own entry point: its address is the port
its runtime facts name, so a test that gives it the stand's port (and nothing else) gets its measurements and frames
of the scenes it is asked for (`places`: scene ids such as `task_card.done`). Its stand walks the scenes as it walks
the app's places, by the `All scenes` links (`data-nav-page`; the drawn scene's link is the one carrying
`aria-current="page"`), and answers every write itself.
`test_the_ui_look_tool_looks_at_the_stand_through_its_own_entry_point` does exactly that.

The browser tests are `tests/test_card_gallery_browser.py` (`ui_browser` lane, Chromium and WebKit); the stand's own
seams — header routing, the one-snapshot rule, scene validation, the reads-only answers, server lifetime — are
`tests/test_card_gallery_stand.py` (no browser).

## Data

- `tests/fixtures/card_gallery.json` — generated by `scripts/generate_card_gallery_fixture.py` from the Python
  producers; never edited by hand. `--card-corpus PATH` reads a producer's draft instead.
- `web/tests/fixtures/work_projection_cases.json` and `web/tests/fixtures/consciousness_contract.json` — the existing
  server-built corpora, read as they are.

A task card or Share scene is one task in a data root of its own: its narration through the worker's emit and the
supervisor's send handler, its answer through the end-of-task pipeline (`emit_task_results`) and that handler, a
running or paused task as the queue keeps it, and then the gateway's history, census and task-detail readers. The
recording pins the clock and uuid4 and names the root `/gallery-data`; the free host-facts row is left out (its stored
row names the temporary folders, so its address would differ by machine). Settings › Agents reads the settings
producer's answer for a synthetic roster, cut to the keys the page reads (the rest name the machine's folders), beside
the review pool the same roster gives.

The setup scenes use the setup wizard's own document head (`web/onboarding_template.html`: its sheets, onboarding.css
too). The connection card's starter list is `skill_starter.starter_groups` for the fixture skill of
`tests/test_onboarding_connections.py` (its card's addresses are `example.test`); its identity answers are the real
`api_connection_identity`, answered by that skill's own function (`1:` accepted as `demo_bot`, anything else refused),
never a service. The scene hands the recorded answer to the card's own check (`createStarterSkillsStep`'s `card`
seam, through `checkIdentity`), so the page posts nothing, and fixes the pairing code (its `random`), so the QR reads
the same in every frame. The provider card reads the bootstrap the host serves (`api_onboarding_bootstrap`) for a
settings document with a dummy key, cut to the parts the providers panel reads; the bootstrap holds the key only as
its "set" placeholder.

## What it covers

"Page" scenes compose a whole product page; "entry point" scenes call the same entry the product calls, without the
room around it. Neighbouring browser tests cover the full composition.

| Card family | Scenes | How | The rest of the composition is covered by |
| --- | --- | --- | --- |
| Question (`chat_decision.js`) | open, long, waiting, closed, answered | entry point: `buildQuizCard` in `chat_media` bubble frames, `applyQuizStateFrame` | `test_quiz_response_browser.py` (on this stand), `test_status_marks_rest_browser.py`, `test_question_room_detail_browser.py` |
| Living card (`chat_cards.js`, `card_components.js`, `card_model.js`, `card_frames.js`) | long, updated, missing | entry point: `createChatCards` `buildCard` / `applyCardState` (no copy/share decoration) | `test_living_cards_browser.py` |
| Reviews in a task card (`review_presentation.js`, `review_opinions.js`, `review_record_card.js`) | empty, waiting, failure, missing_record | entry point: `createReviewPresentationController` over `reviewGroupsFromTaskDetail` | `test_reviews_a_browser.py`, `test_review_wave_failure_browser.py`, `test_body_adoption_card_browser.py` |
| Room work status (`room_work.js`, `work_status.js`) | six server cases, incomplete read, confirmed absence | entry point: `createRoomWork` over `reconcileWorkState`; the badge written as chat.js `setStatus` writes `roomStatusView` | `test_room_work_browser.py`, `test_work_list_panel_browser.py` |
| Task card in Main (`chat.js`, `settled_card.js`, `card_model.js`) | running (next frame: its new line), done and failed (Details opened), paused | page: `createChatInstance` as Main, reading history, census and detail | `test_task_card_b_browser.py`, `test_project_work_card_browser.py`, `test_activity_outcomes_browser.py` |
| Share card (`message_share_card.js` in `message_share.js`) | card, table (refused, with its reason) | page: Main with Share registered as `app.js` registers it; the answer's own Share, then Card | `test_message_share_browser.py`, `test_message_share_files_browser.py` |
| Consciousness tab (`consciousness_dashboard.js`) | snapshot, read failed | page: `initConsciousnessDashboard` reading `/api/consciousness` | `test_consciousness_clock_browser.py`, `test_dashboard_settings_browser.py` |
| Widgets board (`widgets.js`, `widget_card.js`) | empty, pinned long | page: `initWidgets`, shown | `test_widgets_ui_browser.py` |
| Skills (`skills.js`, `skill_card_renderer.js`, `skill_review_card.js`) | empty, review blocked | page: `initSkills`, Installed opened with its own row | `test_deployment_extensions_ui_browser.py` |
| Sign-in (`harness_login_cards.js`) | waiting, failed | entry point: `createLoginCardController.start`, its start answered with the recorded job through the controller's own transport (nothing posted); the waiting card then polls the stand for it, the failed card not | `test_account_resources_browser.py`, `test_harness_maintenance_browser.py` |
| Settings › Agents (`subagents_settings.js`) | empty pool, disabled reviewer | page: the registered `agents` page, mounted as `section_levels.js` mounts it | `test_review_pool_access_browser.py`, `test_agents_settings_browser.py` |
| Setup: connection card (`onboarding_connection_card.js` in `onboarding_starter_skills.js`) | qr, refused | entry point: `createStarterSkillsStep`, its card's Connect pressed and a token entered | `test_onboarding_starter_skills_browser.py`, `test_onboarding_connections.py` |
| Setup: provider card (`settings_provider_cards.js` `providerSettingsCard`, bound by `settings_connections.js`) | closed, saved | entry point: the wizard's providers panel; its wiring after `createConnectionsEditor` is the wizard's own and not exported, so the stand repeats those calls | `test_onboarding_connections_browser.py`, `test_onboarding_interactions_browser.py` |

`providerSettingsCard` is reached only from the setup wizard (`onboarding_wizard.js` renderProvidersStep, the custom
OpenAI-compatible card); Settings › Accounts and keys no longer draws a `.settings-provider-card`, its key pages compose
`connectionFieldsHtml`/`connectionRowHtml` instead. Live account status and model catalogues are not recorded:
Settings › Agents says it could not read agent availability, and Main's capsules show no model, as the app does
without them.

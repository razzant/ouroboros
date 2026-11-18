# Temporary Main and Find preview

This overlay is for one deliberately pushed `qa/costs-and-main-preview` branch in `razzant/ouroboros`. It is not a release gate or a product change. It carries only the synthetic preview fixtures and workflow on a public baseline; no product implementation.

## What runs

- **Main A/B:** public commit `e758a2190298d9890b1f26e632f63775d0b33d55` in a detached runner-temp worktree. The existing loopback SPA fixture serves a synthetic atlas conversation and answers every test action locally. Now demonstrates the room's work list (a held task can also have a Resume there); A uses the stock pill and a System row; B overlays just two responses in its isolated browser page to reveal an original card or an Activity control. Both browsers and both themes cover loaded, older, absent, and held-helper paths, as applicable. Native and mirrored questions also check navigation. The browser asserts that jumping sends no POST and only clicking Resume sends the exact fixture request.
- **Evolution:** the baseline computes its ordinary queued snapshot for two synthetic cycles with the same remaining budget. Only one has an explicit spending pause. The proposal changes the status shown for that case; the dashboard renders both and offers no second Resume button. This is a preview, not an implementation test.
- **Find:** the separate public commit `54f57a5b5baf9c6448bc60a3e7788cf7290dd9be` renders Project and Main with the same fixture in both engines. The mobile viewport is shortened to exercise keyboard geometry; a page-only CSS override is compared with the stock geometry. This is not a physical keyboard test.

The runner installs one locked dependency-only environment and the pinned browser toolchain. The two baseline worktrees must agree with that environment's dependency manifests; otherwise that baseline step fails without replacing or skipping the other preview. Neither preview changes tracked product resources. Each browser check disables credential use by relying on synthetic inputs on the ephemeral runner and by using the existing `safe_test` environment boundary. There is no deployment or credential-backed provider lane.

## What is retained

1. `main-preview-…`: open the standalone `index.html` on disk. It replays screenshots and measured click positions from the browser; it cannot call the application. Frame records and PNGs also ship separately. Missing or interrupted paths are identified in the replay.
2. `main-and-find-checks-…`: the compact lane receipts in `summary.json` (status and counts, plus bounded references to public test files/methods or lines on red), and the Find PNGs with their geometry records.

Raw console, fixture request logs, runner paths and scratch stay private. Python counts distinguish a complete expected failure from a real skip or a missing phase. An engine's replay is labelled complete only after all 30 test cases pass, including browser teardown. Missing checks and nonzero exits stay visible. The Main artifact is uploaded before the Find check. This one run does not substitute for the full browser lane or release checks.

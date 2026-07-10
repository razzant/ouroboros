# Terminal-Bench 2.1 — Methodology

How Ouroboros runs Terminal-Bench 2.1 (harbor) and what a published number means.
Companion to `README.md` (mechanics) — this file is the disclosure/validity SSOT
for TB runs, in the same spirit as `../swe_bench_pro/METHODOLOGY.md`.

## Protocol

- Official harbor harness and task images; `run_tb.py` launches the installed
  full-Ouroboros agent (`harbor_installed_agent.py`) — the agent installs its own
  venv prefix inside each task container (system Python, `pip install -e`, and
  since v6.56.0 `imageio-ffmpeg` in the SAME agent prefix so `extract_video_frames`
  works without touching task-owned packages; a failed install degrades to the
  typed `EXTRACT_VIDEO_FRAMES_UNAVAILABLE` + cv2 workaround hint).
- Agent-setup phase installs AGENT dependencies only — never task dependencies,
  never edits to task files; solving starts only after install completes.
- Harbor owns task timeouts; Ouroboros deadline milestones are inert unless the
  task carries `deadline_at`. The in-container finalization margin is
  `_DEADLINE_SAFETY_SEC=105` (measured overhead, v6.55.0).
- k trials per task follow the leaderboard's k; `disclosure_ledger.json`
  (schema `tb_disclosure_ledger.v1`) records the reward distribution,
  timeout/rate-limit/provider-failure histograms, per-task pass rate,
  concurrency, and the multiplier/gating flags actually used, so each run's
  leaderboard validity is auditable after the fact.

## Review-mode disclosure

The recorded best configuration used BLOCKING task-review. Campaign runs may
deliberately use `advisory` review (e.g. the v6.56.0 gpt/gemini rows) to trade
review latency for wall-clock throughput — that is a DISCLOSED deviation from
the record configuration and the run manifest carries the actual review mode;
numbers from advisory rows must not be presented as the blocking-config record.

## Leaderboard submission

Since 2026-06 the official TB2.1 leaderboard accepts submissions through the
harbor CLI against Harbor Hub (`harbor auth login` → `harbor upload <job_dir>
--public` → `harbor leaderboard submit -l terminal-bench/terminal-bench-2-1`);
the old HF PR flow is a frozen 2.0-only archive. See README "Submitting to the
leaderboard" for the verified mechanics. Methodology-relevant consequences:

- Static validation requires an **ATIF trajectory for every passing trial**
  (`agent/trajectory.json`). The adapter emits it in-container at the end of
  each trial; pre-existing runs are backfilled with
  `build_atif_trajectories.py`. Trajectories are derived verbatim from the
  recorded logs (tool calls, progress narration, final answer, token/cost
  totals) — no synthesis. The Hub additionally runs an LLM dynamic validation
  over these trajectories (reward-hacking review).
- Submitted jobs become **public**. Because campaign runs inject provider
  keys into task containers (disclosed limitation below), the submission copy
  of a job MUST pass `scrub_submission_secrets.py` (structural blanking of
  embedded settings + literal value sweep + zero-leftover verification)
  before upload. Scrubbing only removes credential values; results, traces,
  and disclosure artifacts are uploaded unmodified.
- Numbers submitted to the leaderboard must come from `leaderboard_valid`
  runs (k≥5, no overrides, agent-web off) under the review-mode disclosure
  rules above; advisory-mode rows must not be presented as the blocking
  record.

## Known limitations

- **Ouroboros data root visible to task shells (masking not yet applied).** The
  agent's own data root lives inside the task container and a task shell could
  in principle read it. The bwrap filesystem isolation precedent (GAIA, ef363ff)
  is the candidate fix; applying it to TB is deliberately DEFERRED to its own
  design review (v6.56.0 sprint decision) — TB campaign runs execute WITHOUT the
  mask and this is a disclosed known limitation, mitigated by trace audits
  (reward-hacking sweep over tools.jsonl / solve traces).
- Infra failures (DNS, 429 storms, install timeouts) are classified
  infra-vs-genuine BEFORE scoring; k<5 partials are marked low_k and never
  compared against the leaderboard.

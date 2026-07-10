# GAIA Methodology Notes

GAIA is an answer benchmark, not a code-patch benchmark. This adapter therefore
does not rewrite the scorer or normalize Ouroboros's core `final_answer`.

- **Official score is authoritative.** Use `inspect_evals/gaia` and its
  `gaia_scorer`. `score_gaia.py` may print a local lenient-normalized diagnostic
  to quantify typographic brittleness, but that number is never the headline.
- **Fixed-model Track A.** `settings_base.json` is the committed base template;
  `run_gaia.py` renders a per-run settings file that pins Ouroboros runtime,
  review, and vision model slots to the solve model and disables post-task
  evolution. The default validation model is `google/gemini-2.5-pro`; Sonnet 4.5
  is documented as the later headline comparator against HAL Generalist, not run
  by default here. GAIA permits web browsing; the fixed-model purity boundary is
  whether a *second reasoning model* enters the scaffold.
  - **`quality_openrouter_web` is the default profile for publishable rows.** It
    injects OpenRouter's server-side `openrouter:web_search` tool directly into the
    main solve-model call, so the SAME solve model that reasons also searches — the
    honest parity with OpenAI Codex (searches via gpt-5.5's own native tool) and
    Claude Code (Anthropic server-side `WebSearch`). It stays single-model reasoning
    (no second LLM enters the scaffold); the `web_search` tool is DISABLED in this
    profile so retrieval flows only through the disclosed native path. The search
    engine is recorded per run in the manifest.
  - **`strict_ddgs` is NOT parity** and must not back a headline comparison against
    native-searching harnesses. It pins `OUROBOROS_WEBSEARCH_BACKEND=ddgs`, an
    unofficial DuckDuckGo scraper with no SLA and markedly weaker retrieval — a
    handicap, not a fair measurement, versus Codex/Claude Code. Kept only for a
    no-provider-key ablation.
  - **`web_off_baseline`** disables `web_search` entirely for apples-to-apples
    comparison with older web-off runs.
  Note on the fixed-model `web_search` TOOL (profiles that leave it enabled): its
  backends `openai_responses`/`openrouter_server_tool`/`anthropic_server_tool` issue
  a SEPARATE provider call whose model is `OUROBOROS_WEBSEARCH_MODEL` (default
  `gpt-5.2`) — pin it to the solve model if the tool is enabled, or a second model
  enters the scaffold. `quality_openrouter_web` sidesteps this by disabling the tool
  and searching through the main-loop native path instead.
- **Acceptance review is required.** GAIA Track A measures the full Ouroboros
  scaffold chosen for this sprint: `OUROBOROS_TASK_REVIEW_MODE=required`, empty
  memory, and no post-task evolution. Since v6.55.0 the default worker pool is
  `OUROBOROS_MAX_WORKERS=4` — a DISCLOSED scaffold parameter (recorded per run as
  `worker_scaffold_disclosure` in the manifest). The workers are same-model
  subagent slots for decomposition WITHIN one task; they are never independent
  attempts with selection, so the run stays pass@1. Pass `--max-workers 1` for a
  strict-baseline ablation (the pre-v6.55.0 default, which starved subagent
  decomposition).
- **Safety mode is light in bench templates (v6.55.0).** The solver runs against
  a disposable rendered settings/data root; the LLM safety pass added cost and
  latency without protecting anything the deterministic guards don't cover in
  this context, so bench templates pin `OUROBOROS_SAFETY_MODE=light` (LLM check
  retained for integration tools only; deterministic guards unchanged). User
  defaults are untouched. Note the asymmetry with runtime mode: GAIA stays
  `light` runtime BECAUSE it runs without workspace isolation against a live
  repo — safety-mode light does not weaken that boundary.
- **Runtime mode is light by design.** The accepted plan originally sketched
  `pro`, but review corrected this to `light`: GAIA is an answer benchmark, not
  a self-repo modification task, so the adapter must not give benchmark prompts
  protected Ouroboros repo/control-plane write authority. Light mode still permits
  task/artifact/user-file deliverables needed for answer work while keeping the
  system body protected.
- **Structured extraction.** The solver invokes `ouroboros run
  --result-json-out <sample>/result.json` and reads `final_answer` first, falling
  back to `result` only when the structured field is absent. It does not scrape
  the last stdout line.
- **Answer-format prompt (adapter only).** The solver appends GAIA's standard
  format instruction (a number / as few words as possible / no units unless asked;
  the `FINAL ANSWER:` template), shared as one SSOT constant
  (`inspect_solver.GAIA_FORMAT_INSTRUCTION`) across the Ouroboros/codex/Claude
  solvers. This is GAIA's own intended format/prefix prompt: it shapes the AGENT'S
  OWN answer using only the public task contract, never the gold answer. GAIA's
  quasi-exact-match scorer normalizes whitespace/case/punctuation and selected
  numeric punctuation, but NOT articles, units, scale, or wording, so the format
  prompt is the methodology-sanctioned alignment surface.
  Ouroboros's core `final_answer` and `extract_final_answer` are untouched (a core
  answer-normalizer would harm ordinary users, where units/wording are often part
  of the requested answer).
- **Agent-visible deadline (honesty: visible == real budget − reserve).** GAIA
  imposes no per-task wall-clock limit — the sample timeout is an OPERATOR budget.
  The solver passes `--timeout = GAIA_SAMPLE_TIMEOUT_SEC − reserve` (reserve = 10%,
  capped at 240s) so Ouroboros's existing deadline-awareness (50/25/10% milestones
  + a save-at-10% nudge, `loop.py`) activates and the agent converges to a saved
  answer instead of being killed mid-thought. The visible deadline is STRICTLY
  tighter than the outer hard-kill backstop (`subprocess.run(timeout=…)`), so the
  agent is never told a deadline it is killed before reaching. The deadline conveys
  only time, no answer content. Disclosed here because GAIA is scaffold-sensitive.
- **Attachment access (general runtime capability).** GAIA task files are passed to
  `ouroboros run` via `--attach`; the runtime stages every attachment into the
  task-readable `artifact_store/attachments/` and surfaces a ready-to-read manifest
  (plus native image blocks for images). When Inspect exposes real file paths, the
  adapter passes them directly with `--attach`; when a GAIA prompt still names a
  legacy `/shared_files/...` path and Inspect's TaskState is empty, the adapter
  resolves it under the operator-supplied `GAIA_SHARED_FILES_ROOT` and rewrites the
  prompt toward the attachment manifest. This keeps the runtime prompt consistent
  with the actual sandbox instead of asking the agent to hunt the host filesystem.
- **No best-of-N as pass@1.** One attempt per task is pass@1. Multi-seed or
  pass@k runs must be labelled separately.
- **Historical raw material.** `dragunov_traces/gaia_repro/` remains outside the
  repo as forensic input; useful settings and wrapper ideas are copied/adapted
  here rather than imported as live code.

## Answer-leakage audit protocol (publishable web-on runs)

GAIA's validation answers are public on HuggingFace, so a web-capable agent can
in principle look them up instead of solving the task (UC Berkeley RDI reported
~98% on GAIA by pure answer-lookup, 2026; HAL — Holistic Agent Leaderboard,
arXiv 2510.11977 — surfaces this class via LLM-judge inspection of agent logs,
e.g. "searching for the benchmark on HuggingFace instead of solving"). Our
publishable rows deliberately run WITH web access, because the comparison targets
(OpenAI Codex, Claude Code) are themselves web-using harnesses — a web-off
Ouroboros row would be an unfair handicap, not a fairer measurement. Network
egress is therefore NOT isolated; instead honesty is established by THREE measures:
a prompt-level rule that forbids the lookup, **filesystem isolation** of the answer
cache, and **post-hoc trace auditing** (the HAL-sanctioned alternative to sandboxing)
with a pre-registered scoring rule.

**Filesystem isolation (answer cache).** GAIA's answers are cached on the host
(`metadata.parquet`'s `Final answer` column + per-task `.jsonld`, under
`~/.cache/inspect_evals` and the HuggingFace dataset cache). Every solver runs its
agent as a host subprocess with shell access, so the agent can read that key off disk
(observed 2026-07-04: a Codex row ran `find … gaia_dataset … jq <sample>.jsonld`). Each
solver subprocess — Ouroboros and all three CLI harnesses, symmetrically — is therefore
wrapped in `bwrap` with those answer-cache directories masked by empty tmpfs, leaving the
rest of the filesystem, the network (web search), and the CLI config intact
(`bwrap_isolate.py`, default-on, `GAIA_BWRAP_ISOLATE=0` to disable). The inspect scorer
runs in the main process OUTSIDE the wrapper and reads the dataset normally, so scoring
is unaffected. This is the filesystem complement to the web-side anti-lookup rule.

**Anti-lookup instruction (SSOT, all harnesses).** Every solver appends
`GAIA_ANTI_LEAK_INSTRUCTION` (defined once in `inspect_solver/__init__.py`) to the
task prompt, identically for Ouroboros/Codex/Claude Code/Hermes. It states the
question comes from a public evaluation set whose reference answers are online,
requires deriving the answer from primary sources, and forbids searching for the
evaluation set / its answer files / mirrors / leaderboards. It deliberately does
NOT name the benchmark (so as not to prime the model toward the answer source, and
so echoes of it in traces do not self-trip the audit regex) and does NOT contain
the literal "FINAL ANSWER" marker. **Disclosure:** the pre-2026-07-04 runs did not
carry this instruction, so pre/post-fix rows are not directly comparable.

`audit_leakage.py` implements the audit (diagnostic only — it never changes an
inspect score):

- **Layer 1 — deterministic.** Every web/browser/shell tool call is scanned.
  STRONG flags ("the agent acted on the answer source"): (a) *requested* a leak URL
  (host in `huggingface.co`/`hf.co`/`datasets-server.huggingface.co`/known mirrors,
  or a path naming the GAIA dataset / answer dump — patterns in the SSOT
  `leak_targets.py`), (b) ran an answer-hunting search query
  (`gaia (benchmark|dataset|answers|leaderboard)`, `huggingface … gaia`, or a quoted
  `"FINAL ANSWER"` operator), or (c) retrieved the distinctive gold answer **from a
  leak source** (gold text and a leak URL in the same activity record). WEAK signals
  — a leak URL merely appearing in results, or the gold answer appearing in an
  ordinary page (which is what legitimate research looks like) — are recorded for
  the judge, not hard-flagged. The bare token "gaia"/"GAIA" is never matched against
  raw page text (the ESA Gaia telescope and works named Gaia occur constantly).
- **Layer 2 — LLM judge (HAL-style).** A light model scores each sample's web
  activity against a "benchmark answer search / cheating" rubric. Advisory,
  post-hoc, applied identically to every harness.

**Pre-registered scoring rule (leakage-adjusted headline).** `score_gaia.py
--leakage-audit` reports, alongside the authoritative official accuracy, a
`leakage_adjusted_accuracy` that counts every STRONG-flagged sample as INCORRECT
even if it scored correct (the answer was looked up, not solved). Raw accuracy,
adjusted accuracy, and the flag count are always published together, and the rule
is applied identically to every harness. The official inspect score is never
mutated.

**Per-harness audit coverage (disclose this asymmetry):**
- **Ouroboros** exposes local tool calls (`ouroboros_data/**/tools.jsonl`) AND the
  native server-side web-search citations (`events.jsonl` `web_search_sources`).
  In `quality_openrouter_web` the native path is primary; its citation URLs and
  content are scanned, but the search QUERY text is not logged by the provider, so
  a benchmark-hunting query is invisible for the native path (leak detection there
  is via citation URLs + gold-from-leak, not query text).
- **Codex** runs `codex exec --json`, streaming JSONL tool events to a per-sample
  `codex_trace.jsonl` that the audit scans. Transport disclosure: codex-cli ≥0.142's
  default WebSocket transport to `/v1/responses` does not carry an API key for
  service-account (non-ChatGPT-login) auth, so the solver defines a custom HTTP
  provider (`wire_api=responses`) hitting the SAME direct-OpenAI endpoint and model.
  This is transport-only — it does not relocate or align the endpoint.
- **Claude Code** runs `claude -p --output-format stream-json --verbose`, streaming
  its WebSearch/WebFetch events to a per-sample `claude_code_trace.jsonl` that the
  audit scans (replacing the earlier blind `--output-format json`).
- **Hermes** dumps its verbose tool trace to `hermes_trace.txt`.
  For all CLI traces the appended prompt boilerplate is stripped before scanning so
  an echoed instruction cannot self-flag the sample.

## Hermes baseline (cost-reduced k=1)

The Hermes-agent baseline (NousResearch) is run at reduced sampling for cost:
GAIA at pass@1 like every other row, and Terminal-Bench 2.1 at **k=1** (not the
leaderboard-valid k=5). This is a deliberate budget choice — Hermes is included
as an expected-low reference baseline, not a leaderboard-comparable number. Any
Hermes TB2.1 result is stamped `local_low_k` and must NOT be compared directly to
the k=5 rows; disclose the k asymmetry wherever the number appears.

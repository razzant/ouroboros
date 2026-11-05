# Ouroboros Devtools

`devtools/` contains operator-side and benchmark support code that should be
versioned with Ouroboros without becoming part of the runtime core.

Rules:

- Generated logs, datasets, run outputs, Docker layers, and secrets do not live
  here.
- Choose a benchmark output root outside the source checkout and runtime data;
  use each runner's documented output option or `OUROBOROS_BENCH_RUNS_ROOT`.
- Runtime modules must not import `devtools`.
- This is not an immune-system bypass: touched files are reviewed normally.
- Promote code out of `devtools` only through a separate reviewed runtime plan.

## Test-retirement instruments

DEVELOPMENT "Retiring and merging tests" retires a class of tests by a re-runnable rule. Each tool
states its contract in its module docstring:

- `test_census.py` claims every default-lane test function for the first rule that applies and
  writes `decisions.csv` and `totals.json` to an explicit output directory. Rerun it after a
  target merge to regenerate the candidates.
- `protected_tests.py` regenerates `tests/protected_tests.json`, the set the census subtracts.
- `coverage_census.py` measures per-test coverage on a runner, reduces it there to a small digest,
  and answers locally which deletion candidates cover a live product line no survivor covers.

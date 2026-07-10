## Task context

We want to write the source code for a given executable.
The reference binary is located at `./reference_executable` in the workspace root.

You also have access to the existing documentation bundled in the workspace.

Repository metadata (for orientation only — do not fetch source from the internet):
- instance_id: {{instance_id}}
- repository: {{repository}}
- language: {{language}}
- difficulty: {{difficulty}}

## Your task

Implement the source code to generate an executable of exactly identical behavior as the original.

No project-specific dependencies are pre-installed.
You do NOT have access to the internet.
**IMPORTANT**: Make sure that the executable(s) and everything else that is an artifact is not committed, i.e., is in your `.gitignore` file.
Finally, commit your changes.

Make sure that you have a `./compile.sh` file that produces an executable `./executable` in the workspace root.
`compile.sh` should be executable and should install any dependencies needed to compile the executable.
If your compile.sh fails to compile on a fresh checkout, your task has failed.

### How you are graded (public methodology)

Your submission is graded by a **hidden, parameterized test suite**: each hidden test
runs `./reference_executable` and your `./executable` on a probe invocation the graders
designed (flags, stdin, files, error paths) and requires a **byte-exact match of stdout,
stderr, and the exit code**. Your score is the fraction of hidden tests that pass.
Consequences you should design for:

- "Close enough" output loses the test: a missing trailing newline, different error
  wording, or a different exit code on a bad flag each fail that probe byte-for-byte.
- The hidden probes cover behavior you may never have exercised — the reference's
  observable behavior is the contract, not the subset you happened to test.
- You cannot read the hidden tests; you can only out-probe them (see the matrix below).
- **Program name / `argv[0]` (easy to miss, affects every help/usage/error line).** The
  graders run YOUR binary as `./executable`, but in this workspace you only ever invoke the
  reference as `./reference_executable`. A well-behaved CLI derives its displayed program
  name from `argv[0]` (the basename of the invoked path) — so the reference prints
  `reference_executable` in `Usage:`/help/error output ONLY because that is how it happens
  to be invoked here; the hidden goldens were captured from a binary invoked as `executable`.
  Do NOT hardcode the literal `reference_executable` as your program name — take it from
  `argv[0]` (clap's default `bin_name`, Go `filepath.Base(os.Args[0])`, C `argv[0]`, Python
  `sys.argv[0]`). To check this the way the grader sees it, run your binary under a different
  name (e.g. `cp ./executable /tmp/exe && /tmp/exe --help`) and confirm the program-name
  token follows the invoked name instead of being a frozen `reference_executable`.

### Build requirements (not optional)

`compile.sh` must **build** `./executable` from source you wrote (compile, link, or equivalent).
These are **not** acceptable as the primary build:

- `cp`, `mv`, or `install` of a script or existing file to `./executable` without compilation
- packaging an interpreted script as `./executable` when a compiled implementation in **{{language}}** is feasible in the cleanroom

Prefer a real implementation in **{{language}}** (for example `cc`/`make` for C, `go build` for Go, `cargo build` for Rust) when the reference is a native binary.
Using another language is allowed only when it is a genuine compiled or built artifact and still reproduces behavior from observation — not as a shortcut to avoid implementing the target language.

## Important: This is a reverse-engineering benchmark

Your goal is to write original code from scratch that reproduces the executable's behavior.
The only way to learn what the executable does is to run `./reference_executable` and read its bundled documentation.

Any attempt to obtain source code — whether successful or not — or to wrap/reuse the
provided binary will be detected by an automated judge, disqualified, and scored as zero.

## What is allowed

- Read bundled documentation in the workspace
- Run the reference `./reference_executable` to learn behavior (CLI flags, stdin/stdout, files)
- Write your own tests to probe behavior
- Implement a new solution from scratch in source code
- Use general programming knowledge from pretraining when it does not substitute for observing the reference binary

Key rules:

- Do NOT search the internet, clone repos, or download the project from any package registry
- Do NOT wrap, shim, or delegate to `./reference_executable` or any installed version of the same tool
- Do NOT decompile `./reference_executable` or use strace/ltrace on it (analyzing your own binaries is fine)
- You SHOULD test `./reference_executable` to understand its behavior before writing code.
  If you are dealing with a TUI or an interactive/looping tool, probe it through a PTY
  (python `pty`/`pexpect`) or tmux/libtmux — piped stdin often changes behavior. Confirm
  your clone actually **enters its interactive loop** (prompt appears, accepts multiple
  commands, exits on the same quit conditions) instead of processing one line and exiting.

## Recommended workflow

1. Explore all documentation files in the workspace
2. Play with `./reference_executable` to learn its behavior (CLI flags, stdin/stdout, files, edge cases)
3. Write the source code to implement the behavior in **{{language}}** when practical.
   Write large source files **incrementally, in blocks** (append/extend across several
   edits) instead of emitting one huge single-shot file — a truncated giant write is a
   silent way to lose work.
4. Run `./compile.sh` and build `./executable`
5. **Before committing:** run differential checks (see below) and fix mismatches; repeat until stable
6. Only then commit source, `compile.sh`, and `.gitignore` (not `./executable` or `./reference_executable`)

## Verification before commit (required)

Do **not** commit or finish until you have compared `./executable` against `./reference_executable` on many cases.

At minimum:

1. After `./compile.sh`, run a comparison pass over a **broad probe MATRIX** you designed
   while exploring the reference. Cover the axes the hidden tests cover:
   - every flag/subcommand you discovered × boundary values (0, 1, huge, negative, unicode);
   - stdin-vs-file input for the same data, and the **empty** input case;
   - TTY vs pipe where behavior could differ (see the PTY note above);
   - error paths: bad flags, missing files, malformed input — their exact message and exit code;
   - for interactive tools: that the process ENTERS its loop and handles multiple commands.
   Aim for **at least 20** distinct cases for non-trivial tools (fewer only if the CLI is tiny).
2. For each case, compare **exit code**, **stdout**, and **stderr** between `./reference_executable` and `./executable` **byte-exactly** — that is how the hidden suite compares them. Treat any mismatch as a bug to fix.
3. If output may differ only in inconsequential whitespace, document the exact rule and still minimize differences; do not treat "close enough" as done unless you verified the judge-visible behavior matches.
4. Keep iterating: edit source → `./compile.sh` → re-run the comparison pass → fix failures. Budget time for this loop; it is part of the task.

You may keep probe fixtures and a small `compare_probes.sh` (or similar) in the workspace for this loop. Add probe artifacts and `./executable` to `.gitignore` unless the task explicitly requires committing them.

## What is not allowed

### Obtaining source code

The only source of truth is the reference binary itself and bundled documentation. Do not search
the internet, package registries, or external sources for this project's source code.

### Wrapping or reusing the original binary

Your submission must be a genuine reimplementation. `./reference_executable` is for
observation only — your final `./executable` must not depend on it at runtime.

### Binary analysis of the provided executable

All information about `./reference_executable` must come from normal user interaction (CLI, stdin/stdout).
Do NOT decompile it or use disassemblers, strace, ltrace, or similar on `./reference_executable`.

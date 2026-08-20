"""Editbench: measure Ouroboros file-editing tools on trap-laden editing tasks.

Spawns ONE isolated Ouroboros server (isolated clone of the CURRENT WORKING
TREE, isolated data root, throwaway port), then submits each task N times per
tool configuration, restricting the editing toolset per run via
``disabled_tools``. Grades each run with checker.py and mines token/tool
metrics from the isolated data root.

Configs:
  write_file_only / edit_text_only / apply_patch_only / edit_batch_only
                    — exactly one editing tool available;
  default           — the historical set (write_file + edit_text), i.e. the new
                      tools disabled, so it measures the pre-PR baseline;
  full              — all editing tools available (agent's free choice).

Shell/process/web/delegation tools are disabled in EVERY config so the agent
cannot bypass the editing tool under test (e.g. sed via run_command), and
task review / LLM safety are off so token counts measure editing work only.

Benchmarking the CURRENT WORKING TREE is this launcher's purpose (evaluating
uncommitted editing-tool changes), so real runs on a dirty tree need the
recorded ``--allow-dirty-seed`` escape; the admission gate refuses otherwise.

Usage:
  python devtools/benchmarks/editbench/run_editbench.py --tasks all --runs 3 --allow-dirty-seed
  python devtools/benchmarks/editbench/run_editbench.py --tasks t1_rename --configs full --runs 1 --keep
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter

HERE = pathlib.Path(__file__).resolve().parent
REPO_DIR = HERE.parents[2]
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(HERE))

from devtools.benchmarks.common.manifests import (  # noqa: E402
    admit_benchmark_run,
    finalize_run_manifest,
)
from devtools.benchmarks.common.run_roots import assert_outside_repo, timestamp_run_id  # noqa: E402
from devtools.benchmarks.common.server_runner import (  # noqa: E402
    IsolatedServer,
    _api,
    build_isolated_settings,
    seed_owner_state,
)
from checker import grade_generic  # noqa: E402
from make_fixtures_v2 import T3_NEW_CMDSTR, T3_NEW_STRIP, T3_NEW_SUDO  # noqa: E402

FIXTURES = HERE / "fixtures" / "toyproj"

TASK_PROMPT = """\
In this workspace, rename every MODULE-LEVEL function named exactly `ddd` to `aaa` \
across all Python files: the `def` lines and every reference that resolves to those \
module-level functions (direct calls, `from ... import` names, aliases like `d = models.ddd`, \
and qualified references like `core.ddd`).

Strict scope rules — everything else must remain byte-identical:
- Rename ONLY module-level functions named exactly `ddd` and references to them.
- Class methods named `ddd` are NOT module-level functions: keep them and their call sites unchanged.
- Identifiers that merely contain `ddd` (for example `ddd_helper`, `addd`, `dddx`) stay unchanged.
- Local variables named `ddd` are not functions: keep them unchanged.
- String literals, dict keys, comments, docstrings, and README.md stay unchanged.
- Do not reformat, reorder, or "improve" anything.

When you are done, `python main.py` in the workspace must still print OK. \
Do not create new files and do not commit."""

# Editing tools under test. (edit_sketch was evaluated here and then removed
# from the toolset — see README "v2 takeaways"; its configs are kept out.)
EDIT_TOOLS = ("write_file", "edit_text", "apply_patch", "edit_batch")

# Disabled in EVERY config: escape hatches (shell can sed files; claude_code_edit is a
# whole delegated agent), delegation, web, and misc heavy tools irrelevant to editing.
BASE_DISABLED = [
    "claude_code_edit",
    "run_command", "run_script", "start_service", "stop_service",
    "service_status", "service_logs", "verify_and_record",
    "schedule_subagent", "wait_task", "wait_tasks", "get_task_result",
    "peek_task", "cancel_task", "discard_child_result",
    "web_search", "browse_page", "browser_action", "youtube_transcript",
    "analyze_screenshot", "vlm_query", "view_image", "ocr_pdf", "extract_video_frames",
    "advisory_review", "task_acceptance_review", "request_deep_self_review",
    "plan_task",
]

CONFIGS: dict[str, list[str]] = {
    "write_file_only": [t for t in EDIT_TOOLS if t != "write_file"],
    "edit_text_only": [t for t in EDIT_TOOLS if t != "edit_text"],
    "apply_patch_only": [t for t in EDIT_TOOLS if t != "apply_patch"],
    "edit_batch_only": [t for t in EDIT_TOOLS if t != "edit_batch"],
    "default": ["apply_patch", "edit_batch"],
    "full": [],
}

_V2_CONFIGS = ["edit_text_only", "apply_patch_only", "edit_batch_only", "full"]

T2_PROMPT = """\
In review_state_records.py apply EXACTLY these four changes and nothing else (the file \
must stay byte-identical everywhere else — no reformatting, no other renames):
1. Rename the helper function `_stable_digest` to `_content_digest` — the def and every reference.
2. Rename the helper function `_max_iso_ts` to `_latest_iso_ts` — the def and every reference. \
Do NOT touch `_min_iso_ts`.
3. Change the constant `_MAX_RUN_HISTORY` from 10 to 25.
4. Change the constant `_REVIEW_ATTEMPT_TTL_SEC` from 1800 to 2400.
Do not create new files and do not commit."""

T3_PROMPT = f"""\
In shell_parse.py replace three functions with the new implementations given below. \
Each replacement must be verbatim exactly as given (same indentation), and the rest \
of the file must stay byte-identical. Do not create new files and do not commit.

Replace the function `strip_leading_env_assignments` with:

```python
{T3_NEW_STRIP}
```

Replace the function `shell_command_string` with:

```python
{T3_NEW_CMDSTR}
```

Replace the function `sudo_noninteractive_violation` with:

```python
{T3_NEW_SUDO}
```"""

T4_PROMPT = """\
Refactor: move the function `collect_leading_env` from ouroboros/shell_parse.py \
into ouroboros/git_shell_policy.py as a private helper. Exact spec — everything \
else must stay byte-identical:
1. In ouroboros/shell_parse.py: delete the entire `collect_leading_env` function \
(def line through its final `return`), leaving exactly two blank lines between the \
functions that surrounded it.
2. In ouroboros/git_shell_policy.py: remove `collect_leading_env,` from the \
`from ouroboros.shell_parse import (...)` list (delete that whole line).
3. In ouroboros/git_shell_policy.py: insert the function immediately BEFORE \
`def _git_subcommand_and_args(`, renamed to `_collect_leading_env` (body and \
docstring unchanged, byte-for-byte), with exactly two blank lines before it and \
exactly two blank lines after it.
4. Rename its single call site `collect_leading_env(segment)` to \
`_collect_leading_env(segment)`.
Do not touch ouroboros/utils.py or ouroboros/__init__.py. Do not create new files \
and do not commit."""

T5_PROMPT = """\
In provider_models.py convert string literals from double quotes to single quotes \
per this EXACT rule, and change nothing else:
- Applies to every SINGLE-LINE string literal written with double quotes, including \
prefixed ones (f"...", r"...", b"..." etc.).
- SKIP (leave unchanged): triple-quoted strings/docstrings, and any literal whose \
inner text contains a single quote ('), a double quote, or a backslash.
- Comments are not string literals — never edit comment text.
- The file's code semantics must be unchanged (same AST), only quote characters change.
There are a LOT of eligible literals — be systematic and cover the whole file. \
Do not create new files and do not commit."""

V2 = HERE / "fixtures_v2"
CHECKS = HERE / "checks"

# Task registry: workspace source tree, expected tree, graded files, optional
# behavior check, and the config set to run.
TASKS: dict[str, dict] = {
    "t1_rename": {
        "workspace": FIXTURES,
        "expected": HERE / "fixtures" / "expected",
        "files": ["core.py", "utils.py", "models.py", "config.py", "report.py",
                  "legacy.py", "main.py", "README.md"],
        "prompt": TASK_PROMPT,
        "check": [sys.executable, "main.py"],
        "check_pythonpath": False,
        "configs": list(CONFIGS),
    },
    "t2_surgical": {
        "workspace": V2 / "t2_surgical" / "workspace",
        "expected": V2 / "t2_surgical" / "expected",
        "files": ["review_state_records.py"],
        "prompt": T2_PROMPT,
        "check": None,
        "check_pythonpath": False,
        "configs": _V2_CONFIGS,
    },
    "t3_blocks": {
        "workspace": V2 / "t3_blocks" / "workspace",
        "expected": V2 / "t3_blocks" / "expected",
        "files": ["shell_parse.py"],
        "prompt": T3_PROMPT,
        "check": [sys.executable, str(CHECKS / "t3_check.py")],
        "check_pythonpath": True,
        "configs": _V2_CONFIGS,
    },
    "t4_move": {
        "workspace": V2 / "t4_move" / "workspace",
        "expected": V2 / "t4_move" / "expected",
        "files": ["ouroboros/shell_parse.py", "ouroboros/git_shell_policy.py",
                  "ouroboros/utils.py", "ouroboros/__init__.py"],
        "prompt": T4_PROMPT,
        "check": [sys.executable, str(CHECKS / "t4_check.py")],
        "check_pythonpath": True,
        "configs": _V2_CONFIGS,
    },
    "t5_overhaul": {
        "workspace": V2 / "t5_overhaul" / "workspace",
        "expected": V2 / "t5_overhaul" / "expected",
        "files": ["provider_models.py"],
        "prompt": T5_PROMPT,
        "check": [sys.executable, str(CHECKS / "t5_check.py")],
        "check_pythonpath": True,
        "configs": _V2_CONFIGS + ["write_file_only"],
    },
}


def _log(msg: str) -> None:
    print(f"[editbench] {msg}", flush=True)


def _git(args: list[str], cwd: pathlib.Path) -> tuple[int, str]:
    p = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def _clone_working_tree(run_root: pathlib.Path) -> pathlib.Path:
    """Clone the repo INCLUDING uncommitted working-tree state (diff + untracked)."""
    clone = run_root / "clone"
    rc, out = _git(["clone", "--no-hardlinks", "-q", str(REPO_DIR), str(clone)], run_root)
    if rc != 0:
        raise RuntimeError(f"clone failed: {out}")
    rc, out = _git(["checkout", "-q", "-B", "ouroboros"], clone)
    if rc != 0:
        raise RuntimeError(f"checkout failed: {out}")
    _git(["remote", "remove", "origin"], clone)
    diff = subprocess.run(
        ["git", "diff", "HEAD", "--binary"], cwd=str(REPO_DIR), capture_output=True, text=True
    ).stdout
    if diff.strip():
        p = subprocess.run(["git", "apply", "-"], cwd=str(clone), input=diff, capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"applying working-tree diff failed: {p.stderr}")
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=str(REPO_DIR), capture_output=True, text=True,
    ).stdout.split("\0")
    for rel in [u for u in untracked if u]:
        src = REPO_DIR / rel
        dst = clone / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return clone


def _seed_settings(data_root: pathlib.Path, model: str = "") -> pathlib.Path:
    settings_path = data_root / "settings.json"
    live = pathlib.Path.home() / "Ouroboros" / "data" / "settings.json"
    live_cfg: dict = {}
    if live.exists():
        try:
            live_cfg = json.loads(live.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            live_cfg = {}
    overrides = {
        "OUROBOROS_RUNTIME_MODE": "advanced",
        "OUROBOROS_POST_TASK_EVOLUTION": "false",
        # Keep the measurement about EDITING: no end-of-task review passes, no
        # LLM safety calls (deterministic guards stay), no consciousness lane.
        "OUROBOROS_TASK_REVIEW_MODE": "off",
        "OUROBOROS_SAFETY_MODE": "off",
    }
    if model:
        # Pin the WHOLE main lane, fallbacks included — otherwise a transient
        # failure silently retries on the live fallback model and contaminates
        # the per-model comparison.
        overrides["OUROBOROS_MODEL"] = model
        overrides["OUROBOROS_MODEL_FALLBACKS"] = model
        overrides["OUROBOROS_MODEL_HEAVY"] = ""
    cfg = build_isolated_settings(live_cfg, **overrides)
    cfg.setdefault("TOTAL_BUDGET", 50.0)
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return settings_path


def _make_workspace(run_root: pathlib.Path, name: str, src: pathlib.Path) -> pathlib.Path:
    ws = run_root / "workspaces" / name
    shutil.copytree(src, ws)
    _git(["init", "-q"], ws)
    _git(["add", "-A"], ws)
    _git(["-c", "user.email=editbench@local", "-c", "user.name=editbench",
          "commit", "-q", "-m", "seed workspace"], ws)
    return ws


def _submit(server: IsolatedServer, prompt: str, workspace: pathlib.Path, disabled: list[str], timeout_sec: int) -> str:
    body = {
        "description": prompt,
        "memory_mode": "empty",
        "actor_id": "editbench",
        "source": "editbench",
        "timeout_sec": timeout_sec,
        "workspace_root": str(workspace),
        "workspace_mode": "external",
        "disabled_tools": disabled,
        "metadata": {
            "source": "editbench",
            "delegation_role": "root",
            "disabled_tools": disabled,
        },
    }
    created = _api(server.base_url, "POST", "/api/tasks", body, timeout=60)
    task_id = str(created.get("task_id") or "")
    if not task_id:
        raise RuntimeError(f"task submit failed: {created}")
    return task_id


def _iter_jsonl(path: pathlib.Path):
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except ValueError:
                    continue
    except OSError:
        return


def _matches_task(row: dict, task_id: str) -> bool:
    return task_id in (
        str(row.get("task_id") or ""),
        str(row.get("root_task_id") or ""),
        str(row.get("parent_task_id") or ""),
    )


def _mine_metrics(data_root: pathlib.Path, task_id: str) -> dict:
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0,
             "cache_write_tokens": 0, "cost_usd": 0.0, "llm_calls": 0}
    usage_by_source: dict[str, int] = {}
    seen_usage_rows: set = set()
    for events_file in sorted(set(data_root.rglob("events.jsonl"))):
        for row in _iter_jsonl(events_file):
            if row.get("type") != "llm_usage" or not _matches_task(row, task_id):
                continue
            # Token fields live at the ROW level (pricing.emit_llm_usage_event);
            # review surfaces nest them under "usage". The same row is mirrored
            # into per-task headless events files — dedup by ledger attempt ids.
            u = row.get("usage") if isinstance(row.get("usage"), dict) else row
            key = tuple(row.get("ledger_attempt_ids") or []) or (
                row.get("ts"), row.get("source"), u.get("prompt_tokens"), u.get("completion_tokens"),
            )
            if key in seen_usage_rows:
                continue
            seen_usage_rows.add(key)
            usage["llm_calls"] += 1
            usage["prompt_tokens"] += int(u.get("prompt_tokens") or 0)
            usage["completion_tokens"] += int(u.get("completion_tokens") or 0)
            usage["cached_tokens"] += int(u.get("cached_tokens") or 0)
            usage["cache_write_tokens"] += int(u.get("cache_write_tokens") or 0)
            src = str(row.get("source") or "?")
            usage_by_source[src] = usage_by_source.get(src, 0) + 1
            try:
                usage["cost_usd"] += float(u.get("cost") or 0.0)
            except (TypeError, ValueError):
                pass
    usage["calls_by_source"] = usage_by_source

    tool_calls: Counter[str] = Counter()
    tool_errors: Counter[str] = Counter()
    seen_tool_rows: set[tuple] = set()
    for tools_file in sorted(set(data_root.rglob("tools.jsonl"))):
        for row in _iter_jsonl(tools_file):
            if row.get("type") != "tool_call" or not _matches_task(row, task_id):
                continue
            key = (row.get("ts"), row.get("tool"), row.get("tool_call_id"))
            if key in seen_tool_rows:
                continue  # the same row is mirrored into the budget drive root
            seen_tool_rows.add(key)
            tool = str(row.get("tool") or "?")
            tool_calls[tool] += 1
            if row.get("is_error"):
                tool_errors[tool] += 1

    edit_calls = {t: tool_calls.get(t, 0) for t in EDIT_TOOLS if tool_calls.get(t, 0)}
    edit_errors = {t: tool_errors.get(t, 0) for t in EDIT_TOOLS if tool_errors.get(t, 0)}
    usage["cost_usd"] = round(usage["cost_usd"], 4)
    return {
        "usage": usage,
        "tool_calls": dict(tool_calls),
        "tool_errors": dict(tool_errors),
        "edit_tool_calls": edit_calls,
        "edit_tool_errors": edit_errors,
    }


def _one_run(server: IsolatedServer, run_root: pathlib.Path, data_root: pathlib.Path,
             task_name: str, config: str, idx: int, task_timeout: int) -> dict:
    task = TASKS[task_name]
    name = f"{task_name}.{config}_{idx}"
    ws = _make_workspace(run_root, name, task["workspace"])
    disabled = sorted(set(BASE_DISABLED) | set(CONFIGS[config]))
    started = time.time()
    task_id = _submit(server, task["prompt"], ws, disabled, task_timeout)
    _log(f"{name}: submitted task {task_id}")
    result = server.wait_task(task_id, timeout=task_timeout + 420)
    status = str(result.get("status") or "")
    if status == "timeout":
        server.cancel_task(task_id)
        result = server.wait_task(task_id, timeout=300)
        status = f"timeout->{result.get('status')}"
    wall = round(time.time() - started, 1)
    graded = grade_generic(
        ws, task["expected"], task["files"],
        check_argv=task["check"], check_pythonpath=task["check_pythonpath"],
    )
    metrics = _mine_metrics(data_root, task_id)
    row = {
        "run": name,
        "task": task_name,
        "config": config,
        "task_id": task_id,
        "status": status,
        "wall_sec": wall,
        "pass": graded["pass"],
        "files_matched": graded["files_matched"],
        "files_total": graded["files_total"],
        "behavior_ok": graded["behavior"].get("ok", False),
        "grade": graded,
        **metrics,
    }
    _log(
        f"{name}: status={status} pass={graded['pass']} "
        f"files={graded['files_matched']}/{graded['files_total']} wall={wall}s "
        f"tokens={metrics['usage']['prompt_tokens']}p/{metrics['usage']['completion_tokens']}c "
        f"cost=${metrics['usage']['cost_usd']} edit_calls={metrics['edit_tool_calls']} "
        f"edit_errors={metrics['edit_tool_errors']}"
    )
    return row


def _summarize(rows: list[dict]) -> dict:
    by_config: dict[str, dict] = {}
    for key in sorted({(r.get("task", "t1_rename"), r["config"]) for r in rows}):
        task_name, config = key
        sub = [r for r in rows if r.get("task", "t1_rename") == task_name and r["config"] == config]
        n = len(sub)
        edit_calls = Counter()
        edit_errors = Counter()
        for r in sub:
            edit_calls.update(r["edit_tool_calls"])
            edit_errors.update(r["edit_tool_errors"])
        by_config[f"{task_name}/{config}"] = {
            "runs": n,
            "pass_rate": round(sum(1 for r in sub if r["pass"]) / n, 2),
            "avg_files_matched": round(sum(r["files_matched"] for r in sub) / n, 1),
            "avg_wall_sec": round(sum(r["wall_sec"] for r in sub) / n, 1),
            "avg_prompt_tokens": round(sum(r["usage"]["prompt_tokens"] for r in sub) / n),
            "avg_completion_tokens": round(sum(r["usage"]["completion_tokens"] for r in sub) / n),
            "avg_cost_usd": round(sum(r["usage"]["cost_usd"] for r in sub) / n, 4),
            "avg_llm_calls": round(sum(r["usage"]["llm_calls"] for r in sub) / n, 1),
            "edit_tool_calls": dict(edit_calls),
            "edit_tool_errors": dict(edit_errors),
        }
    return by_config


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", default="t1_rename", help=f"comma list or 'all'; known: {list(TASKS)}")
    ap.add_argument("--configs", default="all", help="comma list, 'all' (per-task set), or 'matrix' (every config)")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--task-timeout", type=int, default=900)
    ap.add_argument("--model", default="", help="pin the main model lane (and its fallbacks), e.g. google/gemini-3.6-flash")
    ap.add_argument("--parallel", type=int, default=1, help="concurrent tasks on the isolated server")
    ap.add_argument("--out", default="", help="results dir (default: <run_root>/results; must be outside the repo)")
    ap.add_argument("--keep", action="store_true", help="keep the temp run root")
    ap.add_argument(
        "--allow-dirty-seed",
        action="store_true",
        help="record and proceed with an unclean seed checkout instead of refusing "
             "(editbench's normal mode: it benchmarks uncommitted editing-tool changes)",
    )
    args = ap.parse_args()

    task_names = list(TASKS) if args.tasks == "all" else [t.strip() for t in args.tasks.split(",") if t.strip()]
    unknown_tasks = [t for t in task_names if t not in TASKS]
    if unknown_tasks:
        _log(f"unknown tasks: {unknown_tasks}; known: {list(TASKS)}")
        return 2
    explicit_configs = None
    if args.configs not in ("all", "matrix"):
        explicit_configs = [c.strip() for c in args.configs.split(",") if c.strip()]
        unknown = [c for c in explicit_configs if c not in CONFIGS]
        if unknown:
            _log(f"unknown configs: {unknown}; known: {list(CONFIGS)}")
            return 2
    jobs: list[tuple[str, str, int]] = []
    for task_name in task_names:
        if explicit_configs is not None:
            task_configs = explicit_configs
        elif args.configs == "matrix":
            task_configs = list(CONFIGS)
        else:
            task_configs = TASKS[task_name]["configs"]
        jobs.extend((task_name, config, idx) for config in task_configs for idx in range(1, args.runs + 1))

    # FAIL FAST on missing fixtures. t2-t5 read `fixtures_v2/`, which is GENERATED
    # from the current tree by make_fixtures_v2.py and is deliberately not committed
    # (it is the tree under measurement, not a pinned snapshot). Discovering that
    # inside copytree() would happen AFTER the isolated server is up and the paid t1
    # jobs have already run — pure argv/path arithmetic belongs before admission.
    missing = sorted({
        str(TASKS[t]["workspace"]) for t, _c, _i in jobs
        if not pathlib.Path(TASKS[t]["workspace"]).is_dir()
    })
    if missing:
        raise SystemExit(
            "editbench: fixture tree(s) not found: " + ", ".join(missing) + "\n"
            "Generate them first: python devtools/benchmarks/editbench/make_fixtures_v2.py"
        )

    # DERIVE the run root, do not CREATE it: admission is the outer boundary, so a
    # seed-gate refusal must leave no filesystem footprint (DEVELOPMENT's
    # admission-as-outer-boundary rule). timestamp_run_id's pid+counter suffix is
    # what keeps two runs started in the same second apart; the atomic manifest
    # write below creates the tree.
    run_root = pathlib.Path(tempfile.gettempdir()) / timestamp_run_id("editbench")
    data_root = run_root / "data"
    out_dir = pathlib.Path(args.out).expanduser() if args.out else run_root / "results"
    out_dir = assert_outside_repo(out_dir, REPO_DIR)
    _log(f"run root: {run_root}")

    # Admission is the outermost refusal point: nothing above spends money or binds ports,
    # so a refused run leaves out_dir holding only the persisted refusal manifest. Editbench
    # exists to benchmark the CURRENT working tree, so a dirty seed is its normal mode — but
    # the escape stays explicit and recorded (--allow-dirty-seed), never implicit.
    manifest_path = out_dir / "run_manifest.json"
    manifest = admit_benchmark_run(
        manifest_path,
        benchmark="editbench",
        run_root=run_root,
        repo_dir=REPO_DIR,
        requested_task_ids=[f"{t}.{c}_{i}" for t, c, i in jobs],
        require_clean=not args.allow_dirty_seed,
        isolated_data_root=str(data_root),
        output_paths={"results": str(out_dir)},
        extra={
            "outcome": "started",
            "tasks": task_names,
            "configs": args.configs,
            "runs_per_config": args.runs,
            "model_pin": args.model or "(live settings default)",
            "parallel": args.parallel,
        },
    )

    with finalize_run_manifest(manifest_path, manifest) as final:
        data_root.mkdir(parents=True)
        clone = _clone_working_tree(run_root)
        settings_path = _seed_settings(data_root, model=args.model)
        if args.model:
            _log(f"main model lane pinned to {args.model}")
        seed_owner_state(data_root)
        from supervisor import state as sstate

        (data_root / sstate.ISOLATED_BENCHMARK_SENTINEL).write_text("isolated benchmark data root\n", encoding="utf-8")

        rows: list[dict] = []
        server = IsolatedServer(clone, data_root, settings_path)
        try:
            _log(f"starting isolated server on {server.base_url} …")
            server.start(ready_timeout=240)
            _log(f"{len(jobs)} runs queued")
            if args.parallel <= 1:
                for task_name, config, idx in jobs:
                    rows.append(_one_run(server, run_root, data_root, task_name, config, idx, args.task_timeout))
                    (out_dir / "runs.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
                    futures = {pool.submit(_one_run, server, run_root, data_root, task_name, config, idx, args.task_timeout): (task_name, config, idx)
                               for task_name, config, idx in jobs}
                    for fut in concurrent.futures.as_completed(futures):
                        rows.append(fut.result())
                        (out_dir / "runs.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        finally:
            server.stop()

        rows.sort(key=lambda r: r["run"])
        summary = _summarize(rows)
        (out_dir / "runs.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        _log("==== SUMMARY ====")
        print(json.dumps(summary, indent=2))
        _log(f"results: {out_dir}")
        final.update({
            "outcome": "completed",
            "exit_code": 0,
            "runs_completed": len(rows),
            "runs_passed": sum(1 for r in rows if r.get("pass")),
        })
        if not args.keep:
            _log("(pass --keep to retain workspaces/logs; keeping run root because results live there)"
                 if not args.out else "cleaning run root")
            if args.out:
                shutil.rmtree(run_root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

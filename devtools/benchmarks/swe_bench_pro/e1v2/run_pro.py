#!/usr/bin/env python3
"""SWE-Pro E1v2 single-root driver with native post-task evolution.

Runs SWE-Pro tasks sequentially against one persistent agent carried by obo-repo/obo-data volumes. Each task is one root task plus one prompt:
  1. solves /app as the ACTIVE EXTERNAL WORKSPACE (v6.56.0: the entrypoint passes
     --workspace /app by default, so contextual repo tools resolve against the task
     repo and a workspace patch artifact is captured; export
     OBO_SOLVE_WORKSPACE_ROOT="" for the legacy rootless user_files mode);
  2. captures the official SWE patch directly from /app (Method C, not --patch-out).
The code-growth channel is native post-task evolution. A workspace task is
project-scoped for per-project FACTS only; global improvement-backlog/promotion
signals still flow (v6.44.0) -> supervisor tick applies promotion -> gated
evolution cycle (reviewed commit in /obo-repo + os.execvpe restart), then wait for absorb, dump state, and continue.
cadence=every_n:1 forces one evolution decision per cycle. Grading is offline.

Modes: default fixed-model baseline (post-task evolution off); --evolution (or the
legacy --self-improve alias) enables native post-task evolution for E1v2 comparison.
--baseline is kept as an explicit compatibility/no-evolution flag.

  OPENROUTER_API_KEY=... python pro/run_pro.py --limit 2 --out-dir runs/pro_smoke --reset-state
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, shutil, subprocess, sys, pathlib, tempfile
# NB: host-wide cache-load serialization uses ouroboros.platform_layer's
# cross-platform file lock (NOT a direct `import fcntl`) so this module stays
# importable on Windows — a bare Unix-only import at any level trips the
# cross_platform checklist gate and breaks the 3-OS `run_pro.py --help` smoke.
from datetime import datetime, timezone

if __package__ in {None, ""}:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))

from devtools.benchmarks.common.run_roots import ensure_outside_repo
from devtools.benchmarks.common.model_slots import pin_single_model

PRO = pathlib.Path(__file__).resolve().parent              # .../swe_bench_pro/e1v2/
ROOT = PRO.parent                                          # .../swe_bench_pro/
SRC = pathlib.Path(__file__).resolve().parents[4]          # Ouroboros repo root (mount ro)
CSV_DEFAULT = ROOT / "task_order_pro_70.csv"
IMG_REPO = "jefzda/sweap-images"


def norm(iid: str) -> str:
    return iid[len("instance_"):] if iid.startswith("instance_") else iid


def read_csv_order(path: pathlib.Path) -> list[str]:
    with path.open(encoding="utf-8") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: int(r["idx"]))
    return [r["instance_id"] for r in rows]


def load_pro_rows(ids: list[str]) -> dict:
    from datasets import load_dataset
    ds = load_dataset("ScaleAI/SWE-bench_Pro", split="test")
    by_key = {}
    for r in ds:
        by_key[r["instance_id"]] = r
        by_key[norm(r["instance_id"])] = r
    out = {}
    for cid in ids:
        row = by_key.get(cid) or by_key.get(norm(cid)) or by_key.get("instance_" + norm(cid))
        if row:
            out[cid] = row
    return out


def read_full_order() -> list[str]:
    from datasets import load_dataset
    return [str(r["instance_id"]) for r in load_dataset("ScaleAI/SWE-bench_Pro", split="test")]


def build_prompt(row: dict, self_improve: bool = True) -> str:
    """Build the solve prompt.

    Uses the clean fixed-version baseline prompt (prompt_baseline.txt): no evolution
    framing, current tool guidance (query_code/search_code/edit in the /app ACTIVE
    WORKSPACE, verify_and_record), and an anti-NOT_EXEC patch-hygiene block. The
    deprecated evolution-mode prompt (prompt_e1v2.txt) is kept for reference only
    (stale: it still describes the legacy user_files dig). E1v2 vs baseline is
    controlled by settings (post-task evolution on/off), not the prompt.
    """
    tpl = (PRO / "prompt_baseline.txt").read_text(encoding="utf-8")
    return (tpl
        .replace("{working_dir}", "/app")
        .replace("{repo}", str(row.get("repo") or ""))
        .replace("{repo_language}", str(row.get("repo_language") or ""))
        .replace("{problem_statement}", str(row.get("problem_statement") or "").strip())
        .replace("{requirements}", str(row.get("requirements") or "").strip())
        .replace("{interface}", str(row.get("interface") or "").strip()))


def build_pretask_evolution_prompt(row: dict, solve_step_budget: int = 0) -> str:
    """Build the owner-controlled pre-task evolution objective for one SWE Pro task.

    Only used under --pretask-evolution (default OFF); the baseline path never
    calls this and never mounts the resulting prompt file.
    """
    step_budget = (
        f"{int(solve_step_budget)} agent steps/rounds"
        if int(solve_step_budget or 0) > 0
        else "the normal SWE Pro agent step/round budget"
    )
    return f"""You are in SWE-bench Pro PRE-TASK SELF-EVOLUTION mode.

The next solve task will run against the repository checked out at /app.
You may inspect /app and the task statement. Your job is to evolve Ouroboros
itself before the solve attempt so the later solve can use the evolved source
and memory.

Important methodology:
- Task-specific evolution is allowed. You may add narrow helpers, workflows,
  repo-specific probes, handoff files, validation commands, or patch-hygiene
  checks for this exact task.
- You are not being given hidden tests, rewards, or answer patches. Do not read
  hidden evaluator files and do not encode hidden answers.
- Do not modify /app as the final task solution during evolution. If you create
  a helper that can later write to /app, it must only do so when explicitly run
  by the solve phase.
- The later solve phase should fit within {step_budget}. There is no separate
  hard wall-clock cutoff for evolution in this benchmark runner, but do not
  waste cycles: evolve capabilities that reduce solve-step waste.
- Shared memory and evolved code persist across the task sequence. Treat this
  as a continuous learning run: use prior evolved capabilities when helpful,
  and improve them rather than duplicating them.

What to produce before stopping:
- one or more reviewed commits when useful;
- a concise handoff for the solve phase, preferably in a discoverable file or
  memory note, with exact commands/workflows to try first;
- visible/local validation signals for the evolved capability;
- a clean worktree, then request_restart so the solve phase loads the evolved
  code.

# Upcoming SWE-bench Pro task
instance_id: {row.get("instance_id", "")}
repo: {row.get("repo", "")}
language: {row.get("repo_language", "")}

<issue>
{str(row.get("problem_statement") or "").strip()}
</issue>

<requirements>
{str(row.get("requirements") or "").strip()}
</requirements>

<interface>
{str(row.get("interface") or "").strip()}
</interface>
"""


def derive_run_settings(base_path: str, out_dir: pathlib.Path, solve_model: str,
                        total_budget: float, per_task_cost: float,
                        post_task_evolution: bool = True, cadence: str = "every_n:1",
                        review_slots: int = 3, review_effort: str = "",
                        runtime_mode: str = "", image_input_mode: str = "") -> pathlib.Path:
    """Build per-run settings for obo-data from the committed base plus benchmark overrides. Secrets are blanked; live keys enter only through explicit environment opt-in."""
    d = json.loads(pathlib.Path(base_path).expanduser().read_text(encoding="utf-8"))
    d["TOTAL_BUDGET"] = float(total_budget)
    d["OUROBOROS_PER_TASK_COST_USD"] = float(per_task_cost)
    # Profile-driven: the passed settings file is the source of truth. We only pin
    # the model slots to --solve-model (a convenience override) and lighten the
    # review triad to --review-slots copies (a single-model run has no reviewer
    # diversity anyway; single_reviewer_no_diversity stays loud). Efforts,
    # runtime_mode, image_input_mode, task_review_mode, etc. flow from the profile
    # unless an explicit override flag is passed.
    if solve_model:
        pin_single_model(solve_model, review_slots=review_slots,
                         review_effort=review_effort, target=d)
    if runtime_mode:
        d["OUROBOROS_RUNTIME_MODE"] = runtime_mode
    if image_input_mode:
        d["OUROBOROS_IMAGE_INPUT_MODE"] = image_input_mode
    d.setdefault("OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS", "true")
    d["OUROBOROS_SERVER_HOST"] = "127.0.0.1"
    d["OUROBOROS_SERVER_PORT"] = 8765
    # cadence "off" disables evolution through the documented POST_TASK_EVOLUTION
    # contract (false), not by relying on a downstream cadence guard — so the CLI's
    # advertised `--cadence off` reliably turns post-task evolution off.
    evolution_enabled = bool(post_task_evolution) and str(cadence).strip().lower() != "off"
    d["OUROBOROS_POST_TASK_EVOLUTION"] = "true" if evolution_enabled else "false"
    d["OUROBOROS_POST_TASK_EVOLUTION_CADENCE"] = cadence
    d["OUROBOROS_POST_TASK_EVOLUTION_BUDGET_USD"] = 0.0
    _STEER_FALLBACK = (
        "At the evolve stage, implement the objective as at most ONE reviewed commit, then restart once. "
        "Fold reviewer fixes into that same change before committing. After the reviewed commit lands "
        "(clean working tree, HEAD = that commit), call request_restart once with a short reason and stop. "
        "An honest no-op is valid when the objective is already solved, unsafe, too broad, or needs owner input. "
        "Do not churn. Do ABSOLUTELY NO release bookkeeping in this benchmark environment: never edit VERSION, "
        "CHANGELOG, README, docs/ARCHITECTURE, pyproject.toml, or package.json, and do not apply any version-bump / "
        "P9 release-carrier rule; advisory review will flag their absence, which is expected and must be left as "
        "advisory. Never modify the review-enforcement machinery to make findings always block or pass regardless "
        "of the configured mode."
    )
    try:
        _steer = (PRO / "prompt_evolution_steer.txt").read_text(encoding="utf-8").strip()
    except Exception:
        _steer = ""
    d["OUROBOROS_EVOLUTION_PERSISTENT_OBJECTIVE"] = _steer or _STEER_FALLBACK
    for k in list(d):
        if any(t in k.upper() for t in ("API_KEY", "TOKEN", "PASSWORD", "SECRET")):
            d[k] = ""
    p = out_dir / "_run_settings.json"
    p.write_text(json.dumps(d, indent=2), encoding="utf-8")
    return p


def read_spent_usd(vdata: str = "obo-data") -> float:
    # alpine:3 (pre-pulled), not the task image: the task image may already be
    # evicted by the per-shard cleanup, and an implicit multi-GB re-pull here would
    # stall every task boundary for the 180s timeout.
    try:
        r = subprocess.run(["docker", "run", "--rm", "--pull=never", "-v", f"{vdata}:/d:ro",
                            "--entrypoint", "cat", "alpine:3", "/d/state/state.json"],
                           capture_output=True, text=True, timeout=180)
        return float(json.loads(r.stdout or "{}").get("spent_usd", 0.0))
    except Exception:
        return 0.0


def kill_container(name: str) -> None:
    try:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=120)
    except Exception:
        pass


def volume_exists(name: str) -> bool:
    return subprocess.run(["docker", "volume", "inspect", name], capture_output=True).returncode == 0


def image_libc(img: str) -> str | None:
    """Choose the environment volume that matches the task image libc (glibc versus musl).

    Returns None when the probe could not run at all (image missing / daemon error):
    defaulting to "glibc" there would silently hand a musl task the glibc env volume
    and record a never-executed task as a genuine 0-byte failure. `--pull=never`
    keeps a concurrently-evicted image from triggering an implicit Hub pull.
    """
    try:
        r = subprocess.run(["docker", "run", "--rm", "--pull=never", "--entrypoint", "sh", img, "-c",
                            "ls /lib/libc.musl* >/dev/null 2>&1 && echo musl || echo glibc"],
                           capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            return None
        return "musl" if "musl" in (r.stdout or "") else "glibc"
    except Exception:
        return None


def dump_state(out: pathlib.Path, vrepo: str = "obo-repo", vdata: str = "obo-data",
               vsuf: str = "") -> None:
    # Name the teardown containers `obopro<suffix>-dump-*` so auto_run's TimeoutExpired
    # handler (which removes `name=obopro<suffix>-`) can reap a dump that hangs under a
    # loaded docker daemon without touching other shards' containers.
    # alpine:3 (pre-pulled), not the task image: the task image may be evicted by the
    # per-shard cleanup between container exit and this dump, and an implicit re-pull
    # would hang the teardown into the task-wall-timeout.
    base = f"obopro{vsuf}-dump-" + out.name.replace("/", "-").replace("_", "-").lower()[:50]
    for vol, name in ((vdata, "obo-data.tgz"), (vrepo, "obo-repo.tgz")):
        try:
            subprocess.run(
                ["docker", "run", "--rm", "--pull=never", "--name", f"{base}-{vol}",
                 "-v", f"{vol}:/src:ro", "-v", f"{out}:/dump",
                 "--entrypoint", "tar", "alpine:3", "czf", f"/dump/{name}", "-C", "/src", "."],
                capture_output=True, timeout=1200)
            sz = (out / name).stat().st_size if (out / name).exists() else 0
            print(f"[pro]   dump {name}: {sz//1024} KiB", file=sys.stderr)
        except Exception as e:
            print(f"[pro]   dump {name} FAILED: {e}", file=sys.stderr)


# Host image cache (docker save | zstd). Default OFF; the re-run sets OBO_SWEPRO_IMG_CACHE to a
# roomy host dir (e.g. on the 3TB host disk, NOT colima's ~197GB VM) so a pulled image is saved
# once and future re-runs load it locally instead of re-pulling ~GBs over the network. The READ
# (cache-load) path still honors a present legacy dir for back-compat; only the WRITE (populate)
# path is gated on the explicit opt-in, so behavior is unchanged when the env is unset.
_IMG_CACHE_ENV = os.environ.get("OBO_SWEPRO_IMG_CACHE", "")
IMG_CACHE = pathlib.Path(_IMG_CACHE_ENV or "/Volumes/OBOCACHE/swebench-cache")
_CACHE_WRITE_ENABLED = bool(_IMG_CACHE_ENV)


def _cache_path(img: str) -> pathlib.Path:
    # `[-1]`: a tagless image (no `:`) yields the whole name instead of an IndexError.
    return IMG_CACHE / f"sweap_{img.split(':', 1)[-1].replace('/', '_')}.tar.zst"


def _save_image_to_cache(img: str) -> None:
    """Populate the host image cache (``docker save | zstd``) so a future re-run loads the image
    locally instead of re-pulling. Gated on the OBO_SWEPRO_IMG_CACHE opt-in. Fail-soft (skip if
    ``zstd`` is missing or the dir is unusable), 'existing valid cache wins' (never clobber), and
    atomic (tmp + os.replace) so a concurrent/interrupted save never leaves a corrupt cache file."""
    if not _CACHE_WRITE_ENABLED:
        return
    tmp = None
    try:
        if shutil.which("zstd") is None:
            print("[pro] cache-save skipped: zstd not found", file=sys.stderr)
            return
        cp = _cache_path(img)
        if cp.is_file() and cp.stat().st_size > 1_000_000:
            return  # existing valid cache wins
        IMG_CACHE.mkdir(parents=True, exist_ok=True)
        tmp = cp.with_name(cp.name + f".tmp.{os.getpid()}")
        dp = subprocess.Popen(["docker", "save", img], stdout=subprocess.PIPE)
        try:
            with open(tmp, "wb") as fh:
                rc = subprocess.run(["zstd", "-q"], stdin=dp.stdout, stdout=fh, timeout=3600).returncode
        finally:
            # Always reap the docker-save child, even if zstd raised/timed out (Process Custody).
            if dp.stdout:
                dp.stdout.close()
            try:
                dp.wait(timeout=60)
            except Exception:
                dp.kill()
                dp.wait(timeout=10)
        if rc == 0 and dp.returncode == 0 and tmp.stat().st_size > 1_000_000:
            os.replace(str(tmp), str(cp))  # atomic publish
            print(f"[pro] cached {cp.name} ({cp.stat().st_size / 1e9:.2f}GB)", file=sys.stderr)
        else:
            tmp.unlink(missing_ok=True)
    except Exception as e:  # noqa: BLE001 — cache population is best-effort, never fail the run
        print(f"[pro] cache-save skipped ({e})", file=sys.stderr)
        if tmp is not None:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass


def _image_present(img: str) -> bool:
    # Timed `docker image inspect`: a wedged docker daemon (e.g. colima under heavy
    # concurrent load) must not block the orchestrator indefinitely here.
    try:
        return subprocess.run(["docker", "image", "inspect", img],
                              capture_output=True, timeout=60).returncode == 0
    except Exception:
        return False


UTIL_IMAGE = "alpine:3"


def ensure_util_image(img: str = UTIL_IMAGE) -> None:
    """Guarantee the small utility image is present locally BEFORE any code path
    uses it with `--pull=never` (state reads, snapshot/dump, restore). This is the
    ONE place it is pulled: a missing util image would otherwise silently degrade
    to a $0 spend read (dead cumulative-budget gate) and empty-volume restores, so
    fail LOUD here instead. No-op when already present (the steady state)."""
    if _image_present(img):
        return
    print(f"[pro] util image {img} absent — pulling once", file=sys.stderr)
    try:
        rc = subprocess.run(["docker", "pull", img], capture_output=True, timeout=300).returncode
    except Exception as e:  # noqa: BLE001 - report and fail closed below
        rc = -1
        print(f"[pro] util image pull error: {e}", file=sys.stderr)
    if rc != 0 or not _image_present(img):
        raise RuntimeError(
            f"utility image {img!r} is unavailable and could not be pulled; snapshot/"
            f"restore/state-read all run with --pull=never and would silently corrupt "
            f"run state (empty-volume restores, $0 spend reads). Pull it manually "
            f"(`docker pull {img}`) and retry.")


def docker_pull_if_missing(img: str) -> bool:
    """Make `img` locally available (cache load, then registry pull). Returns presence.

    Never raises: an uncaught TimeoutExpired from the fallback pull would kill the
    whole run_pro invocation (no timeline row -> auto_run retries the task blind,
    burning hours). Callers turn False into an explicit `image_unavailable` infra
    result instead.
    """
    if _image_present(img):
        return True
    cp = _cache_path(img)
    if cp.is_file() and cp.stat().st_size > 1_000_000:
        # Serialize decompress+load host-wide to <=2 concurrent (slot by tag hash):
        # 12 shards racing multi-GB zstd loads saturate the shared array and blow
        # the load timeout for everyone.
        from ouroboros import platform_layer  # cross-platform file lock (guards fcntl internally)
        slot = int(hashlib.sha1(img.encode()).hexdigest(), 16) % 2
        lock_path = pathlib.Path(tempfile.gettempdir()) / f"obo_swepro_imgload_{slot}.lock"
        zp = None
        with open(lock_path, "w") as lock_fh:
            platform_layer.file_lock_exclusive(lock_fh.fileno())
            print(f"[pro] load from cache {cp.name} ({cp.stat().st_size/1e9:.2f}GB)", file=sys.stderr)
            try:
                zp = subprocess.Popen(["zstd", "-dc", str(cp)], stdout=subprocess.PIPE)
                subprocess.run(["docker", "load"], stdin=zp.stdout, timeout=3600)
                if zp.stdout:
                    zp.stdout.close()
                zp.wait(timeout=60)
                if _image_present(img):
                    return True
                print("[pro] cache-load produced no image - fallback to pull", file=sys.stderr)
            except Exception as e:
                print(f"[pro] cache-load failed/timed out ({e}) - fallback to pull", file=sys.stderr)
            finally:
                # Never leak the decompressor child on any failure/timeout path.
                if zp is not None and zp.poll() is None:
                    try:
                        zp.kill(); zp.wait(timeout=10)
                    except Exception:
                        pass
    print(f"[pro] pull {img}", file=sys.stderr)
    try:
        subprocess.run(["docker", "pull", img], timeout=3600)
    except Exception as e:
        print(f"[pro] pull failed/timed out ({e})", file=sys.stderr)
    if not _image_present(img):
        return False
    _save_image_to_cache(img)  # populate the host cache so future re-runs don't re-pull
    return True


def run_instance(cid: str, row: dict, args, api_key: str, seed_settings: pathlib.Path,
                 task_total: float) -> dict:
    out = (ensure_outside_repo(pathlib.Path(args.out_dir).expanduser(), SRC) / cid.replace("/", "_")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "problem_statement.txt").write_text(build_prompt(row, args.self_improve), encoding="utf-8")
    pretask_on = bool(getattr(args, "pretask_evolution", False))
    if pretask_on:
        (out / "pretask_evolution_prompt.txt").write_text(
            build_pretask_evolution_prompt(row, getattr(args, "solve_step_budget", 0)),
            encoding="utf-8",
        )
    img = f"{IMG_REPO}:{row['dockerhub_tag']}"
    libc = image_libc(img) if docker_pull_if_missing(img) else None
    if libc is None:
        # Image absent after cache-load + pull (or probe could not run): the task never
        # executed. Non-permanent infra (a retry can re-load from the host cache).
        print(f"[pro] {cid}: SOLVE_INFRA_SUSPECT reason=image_unavailable", file=sys.stderr)
        return {"instance_id": cid, "model_name_or_path": args.model_name, "model_patch": "",
                "timed_out": False, "infra_suspect": True, "health_rollback": False,
                "infra_reason": "image_unavailable", "image_ref": img,
                "refl_line": "", "solve_line": "", "quiet_line": ""}
    env_vol = "oboros-env" if libc == "glibc" else "oboros-env-musl"
    install_in_image = False
    if not volume_exists(env_vol):
        if libc == "musl":
            # No musl conda env volume (musllinux wheels for tree-sitter et al. are
            # unreliable). Install Ouroboros INTO the Alpine task image at container
            # start instead — the Terminal-Bench install-in-image transport, with a
            # graceful degrade without tree-sitter. glibc still uses the prebuilt volume.
            install_in_image = True
            print(f"[pro] {cid}: musl image, no '{env_vol}' -> install-in-image transport", file=sys.stderr)
        else:
            print(f"[pro] {cid}: SKIP - missing env volume '{env_vol}' for libc={libc}", file=sys.stderr)
            return {"instance_id": cid, "model_name_or_path": args.model_name, "model_patch": "",
                    "timed_out": False, "infra_suspect": True, "health_rollback": False,
                    "infra_reason": "libc_skip",
                    "libc_skip": f"{libc}:{env_vol}", "refl_line": "", "solve_line": "", "quiet_line": ""}
    if str(os.environ.get("OUROBOROS_BENCH_ALLOW_CONTAINER_SECRETS", "")).lower() not in {"1", "true", "yes"}:
        print("[pro] refusing to inject the provider key into an untrusted Pro task container; set OUROBOROS_BENCH_ALLOW_CONTAINER_SECRETS=1 for audited local smoke only", file=sys.stderr)
        return {"instance_id": cid, "model_name_or_path": args.model_name, "model_patch": "",
                "timed_out": False, "infra_suspect": True, "health_rollback": False,
                "secret_opt_in_required": True, "refl_line": "", "solve_line": "", "quiet_line": ""}
    cname = "obopro" + (getattr(args, "volume_suffix", "") or "") + "-" + norm(cid).replace("__", "-").replace("_", "-").replace(".", "-").lower()[:84]
    M = lambda h, c, ro=True: ["-v", f"{h}:{c}" + (":ro" if ro else "")]
    # Pre-task evolution wiring is fully flag-gated: with --pretask-evolution off
    # (the default) the docker command, mounts, and container env are unchanged.
    pretask_flags = []
    if pretask_on:
        pretask_flags = [
            "-e", "OBO_PRETASK_EVOLUTION=1",
            "-e", f"OBO_PRETASK_EVOLUTION_WAIT_MAX={getattr(args, 'pretask_evolution_wait_max', 0)}",
            "-e", f"OBO_SOLVE_STEP_BUDGET={getattr(args, 'solve_step_budget', 0)}",
            *M(out / "pretask_evolution_prompt.txt", "/opt/pretask_evolution_prompt.txt"),
        ]
    mem_flags = []
    if args.mem_limit:
        # --memory-swap == --memory disables swap so a runaway allocation is
        # capped at the RAM limit (clean OOM, exit 137) rather than swapping the
        # host to death. See README "Diagnosing SIGKILL / OOM".
        mem_flags = ["--memory", args.mem_limit, "--memory-swap", args.mem_limit]
    cmd = ["docker", "run", "--rm", "--pull=never", "--name", cname,
        *mem_flags,
        # Name-only env form: docker forwards the value from our process environment
        # (set below) so the live key never appears in the host argv / `ps` output.
        "-e", "OPENROUTER_API_KEY",
        # Direct-OpenAI solve models (openai::gpt-5.5) route to api.openai.com and need
        # OPENAI_API_KEY; forward it name-only (value via docker_env=os.environ below) when set.
        *(["-e", "OPENAI_API_KEY"] if os.environ.get("OPENAI_API_KEY", "").strip() else []),
        "-e", f"OUROBOROS_MODEL={args.solve_model}",
        "-e", f"OUROBOROS_MODEL_HEAVY={args.solve_model}",
        "-e", f"OUROBOROS_MODEL_LIGHT={args.solve_model}",
        "-e", f"OUROBOROS_MODEL_FALLBACKS={args.solve_model}",
        # Runtime mode flows from the generated settings profile (seed settings.json);
        # only force it via env when --runtime-mode is explicitly set, so a profile's
        # mode is never silently overridden. Committed profiles all carry pro
        # (v6.55.0 container-bench default); pass --runtime-mode light for an ablation.
        *(["-e", f"OUROBOROS_RUNTIME_MODE={args.runtime_mode}"] if args.runtime_mode else []),
        "-e", "OUROBOROS_PRE_PUSH_TESTS=0",
        "-e", f"TOTAL_BUDGET={task_total}",
        "-e", f"OUROBOROS_PER_TASK_COST_USD={args.per_task_cost}",
        "-e", f"OBO_BASE_COMMIT={row['base_commit']}",
        "-e", f"OBO_INSTANCE_ID={cid}",
        "-e", f"OBO_REPO={row.get('repo','')}",
        "-e", "OBO_WORKDIR=/app",
        "-e", f"OBO_SOLVE_TIMEOUT={args.solve_timeout}",
        "-e", f"OBO_ABSORB_MAX={args.absorb_max}",
        "-e", f"OBO_REFLECT_MIN={args.reflect_min}",
        "-e", f"OBO_REFLECT_MAX={args.reflect_max}",
        "-e", f"OBO_QUIET_STABLE={args.quiet_stable}",
        "-e", "OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS=true",
        "-e", f"OBO_SELFIMPROVE={1 if args.self_improve else 0}",
        "-e", "OUROBOROS_MAX_SUBAGENT_DEPTH=2",
        "-e", "OUROBOROS_MAX_ACTIVE_SUBAGENTS_PER_ROOT=3",
        "-e", "OUROBOROS_SUBAGENT_WORKTREE_ROOT=/Ouroboros/subagent_worktrees",
        "-e", f"OBO_MEMORY_MODE={args.memory_mode}",
        "-e", f"OBO_DISABLE_TOOLS={args.disable_tools}",
        # Workspace-root passthrough (v6.56.0): the entrypoint defaults to /app as
        # the active external workspace; forward an explicit host override (incl.
        # the EMPTY string = legacy rootless user_files mode) only when set.
        *(["-e", f"OBO_SOLVE_WORKSPACE_ROOT={os.environ['OBO_SOLVE_WORKSPACE_ROOT']}"]
          if "OBO_SOLVE_WORKSPACE_ROOT" in os.environ else []),
        "-e", f"OBO_INSTALL_IN_IMAGE={1 if install_in_image else 0}",
        # glibc mounts the prebuilt conda env volume read-only; musl install-in-image
        # builds a venv inside the task image instead (no volume mounted).
        *([] if install_in_image else ["-v", f"{env_vol}:/opt/miniconda3/envs/oboros:ro"]),
        "-v", f"obo-repo{(getattr(args, 'volume_suffix', '') or '')}:/obo-repo", "-v", f"obo-data{(getattr(args, 'volume_suffix', '') or '')}:/obo-data",
        *M(SRC, "/opt/ouroboros-ro"),
        *M(seed_settings, "/opt/oboros-settings-ro.json"),
        *M(PRO / "entrypoint_pro.sh", "/opt/entrypoint_pro.sh"),
        *M(out / "problem_statement.txt", "/opt/problem_statement.txt"),
        *pretask_flags,
        "-v", f"{out}:/out",
        "--entrypoint", "bash", img, "/opt/entrypoint_pro.sh"]
    kill_container(cname)
    timed_out = False
    oom = False
    host_to = args.solve_timeout + args.absorb_max + 1200
    # Pass the provider key through the child environment (not argv) for the
    # name-only `-e OPENROUTER_API_KEY` above.
    docker_env = dict(os.environ)
    docker_env["OPENROUTER_API_KEY"] = api_key
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=(None if getattr(args, "no_hard_timeouts", False) else host_to), env=docker_env)
        oom_note = ""
        if r.returncode == 137:
            oom = True
            oom_note = (
                f"\n[driver] container exited 137 (SIGKILL) — likely OOM at --mem-limit={args.mem_limit}. "
                "A worker killed mid-task with 'signal 9 — terminal' is usually this. Check for an "
                "unbounded operation (e.g. search over a root that resolved to '/'); raise --mem-limit "
                "or narrow the task. Host dmesg shows the kernel OOM line.\n"
            )
        img_note = ""
        if r.returncode != 0 and "No such image" in (r.stderr or ""):
            # --pull=never: image evicted between the libc probe and this run.
            img_note = "\nSOLVE_INFRA_SUSPECT reason=image_unavailable (--pull=never: image gone before run)\n"
        (out / "container.log").write_text(r.stdout + "\n" + r.stderr + oom_note + img_note, encoding="utf-8")
    except subprocess.TimeoutExpired as e:
        timed_out = True
        dec = lambda b: b.decode("utf-8", "replace") if isinstance(b, bytes) else (b or "")
        (out / "container.log").write_text("[driver] CONTAINER TIMEOUT\n" + dec(e.stdout) + "\n" + dec(e.stderr), encoding="utf-8")
        kill_container(cname)
        print(f"[pro] {cid}: TIMEOUT (host) - continuing", file=sys.stderr)

    patch = (out / "patch.diff").read_text(encoding="utf-8") if (out / "patch.diff").exists() else ""
    clog = (out / "container.log").read_text(encoding="utf-8", errors="replace") if (out / "container.log").exists() else ""
    ilog = (out / "install.log").read_text(encoding="utf-8", errors="replace") if (out / "install.log").exists() else ""
    se = out / "solve_events.jsonl"
    api_net = api_ctx = 0
    n_events = 0
    _CTX = ("prompt is too long", "input is too long", "context length",
            "maximum context", "context_length", "too many tokens")
    if se.exists():
        for ln in se.read_text(errors="replace").splitlines():
            if ln.strip():
                n_events += 1
            if '"llm_api_error"' not in ln:
                continue
            try:
                ev = json.loads(ln)
            except Exception:
                continue
            if ev.get("type") != "llm_api_error":
                continue
            err = str(ev.get("data", {}).get("error", "")).lower()
            if any(t in err for t in _CTX):
                api_ctx += 1
            else:
                api_net += 1
    api_errors = api_net   # gate-relevant count = network transients, not context overflow
    def grep1(marker: str) -> str:
        for ln in clog.splitlines():
            if marker in ln:
                return ln.strip()
        return ""
    infra_reason = ""
    for line in (clog + "\n" + ilog).splitlines():
        if "SOLVE_INFRA_SUSPECT reason=" in line:
            infra_reason = line.split("SOLVE_INFRA_SUSPECT reason=", 1)[1].split()[0].strip()
            break
    if not infra_reason and "SOLVE_INFRA_SUSPECT" in (clog + "\n" + ilog):
        infra_reason = "infra_suspect"
    selfedit = {}
    sep = out / "selfedit.json"
    if sep.exists():
        try:
            selfedit = json.loads(sep.read_text(encoding="utf-8"))
        except Exception:
            selfedit = {}
    absorb = {}
    abp = out / "absorb.json"
    if abp.exists():
        try:
            absorb = json.loads(abp.read_text(encoding="utf-8"))
        except Exception:
            absorb = {}
    image_id = ""
    try:
        image_id = subprocess.run(["docker", "image", "inspect", "-f", "{{.Id}}", img],
                                  capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception:
        pass
    res = {"instance_id": cid, "model_name_or_path": args.model_name, "model_patch": patch,
           "timed_out": timed_out, "api_errors": api_errors, "api_ctx": api_ctx,
           "oom": oom, "n_events": n_events, "image_ref": img, "image_id": image_id,
           "infra_suspect": "SOLVE_INFRA_SUSPECT" in (clog + "\n" + ilog),
           "infra_reason": infra_reason,
           "health_rollback": ("HEALTH_GATE_ROLLBACK" in clog) or ("PRETASK HEALTH-GATE FAILED" in clog),
           "selfedit": selfedit,
           "evolution_degraded": bool(absorb.get("degraded")),
           "absorb_reason": str(absorb.get("reason", "")),
           "refl_line": grep1("knowledge files:"),
           "solve_line": grep1("ROOT-RUN patch="),
           "quiet_line": grep1("[pro] evolution:")}
    if pretask_on:
        pretask = {}
        pep = out / "pretask_evolution.json"
        if pep.exists():
            try:
                pretask = json.loads(pep.read_text(encoding="utf-8"))
            except Exception:
                pretask = {}
        res["pretask_evolution"] = pretask
    return res


def normalize_result(row: dict, cid: str, args) -> dict:
    defaults = {
        "instance_id": cid,
        "model_name_or_path": args.model_name,
        "model_patch": "",
        "timed_out": False,
        "infra_suspect": False,
        "health_rollback": False,
        "infra_reason": "",
        "api_errors": 0,
        "api_ctx": 0,
        "oom": False,
        "n_events": 0,
        "image_ref": "",
        "image_id": "",
        "refl_line": "",
        "solve_line": "",
        "quiet_line": "",
        "selfedit": {},
        "evolution_degraded": False,
        "absorb_reason": "",
    }
    return {**defaults, **(row or {})}


def resume_result(cid: str, cid_dir: pathlib.Path, model_name: str) -> dict | None:
    """Rebuild a task result from an already-captured patch WITHOUT touching Docker.

    A prior (possibly teardown-killed) invocation may have left a non-empty
    ``patch.diff`` for this task. Resuming must NOT re-pull the image or read state
    via ``docker run`` (that would reintroduce the image-pull stall this hardening
    removes), so the resume path reads only local files. Returns the result dict, or
    None when there is no usable captured patch.
    """
    p = cid_dir / "patch.diff"
    try:
        if not (p.exists() and p.stat().st_size > 0):
            return None
        return {"instance_id": cid, "model_name_or_path": model_name,
                "model_patch": p.read_text(encoding="utf-8", errors="replace")}
    except OSError:
        return None


def build_timeline_row(order: int, cid: str, res: dict, spent_after: float, flags: list) -> dict:
    """Build one timeline.jsonl row.

    Persists the infra non-execution markers (`infra_suspect`,
    `secret_opt_in_required`, `libc_skip`) so auto_run.run_one can hard-stop on a
    secret-injection refusal and avoid counting a skipped/non-executed task as a
    LEGIT last-good. Dropping them here would silently re-break that handoff.
    """
    se = res.get("selfedit") or {}
    return {"order": order, "instance_id": cid, "patch_bytes": len(res["model_patch"]),
            "ts": datetime.now(timezone.utc).isoformat(),
            "spent_after_usd": round(spent_after, 4), "flags": flags,
            "infra_suspect": bool(res.get("infra_suspect")),
            "infra_reason": str(res.get("infra_reason") or ""),
            "secret_opt_in_required": bool(res.get("secret_opt_in_required")),
            "libc_skip": res.get("libc_skip", ""),
            "api_errors": res["api_errors"], "api_ctx": res["api_ctx"],
            "oom": bool(res.get("oom")), "n_events": int(res.get("n_events", 0)),
            "image_ref": str(res.get("image_ref", "")), "image_id": str(res.get("image_id", "")),
            "refl": res["refl_line"], "quiet": res["quiet_line"],
            "commits_added": se.get("commits_added", 0),
            "loc_added": se.get("loc_added", 0), "loc_removed": se.get("loc_removed", 0),
            "tools_added": se.get("tools_added", []), "verdicts": se.get("verdicts", {}),
            "self_rollback": se.get("health_rollback", False),
            # Present only under --pretask-evolution; baseline timeline rows are unchanged.
            **({"pretask_evolution": res["pretask_evolution"]} if "pretask_evolution" in res else {}),
            "evolution_degraded": res.get("evolution_degraded", False),
            "absorb_reason": res.get("absorb_reason", "")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(CSV_DEFAULT))
    ap.add_argument("--full-set", action="store_true", help="run the full HF ScaleAI/SWE-bench_Pro test split instead of the CSV order")
    ap.add_argument("--start", type=int, default=1, help="first task index (1-based, from CSV)")
    ap.add_argument("--limit", type=int, default=2, help="number of tasks to run")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--settings", default=str(PRO / "settings_base.json"),
                    help="benchmark base settings.json (committed template, not a personal agent folder)")
    ap.add_argument("--solve-model", default="openai/gpt-5.5")
    ap.add_argument("--total-budget", type=float, default=500.0)
    ap.add_argument("--per-task-cost", type=float, default=25.0)
    ap.add_argument("--mem-limit", default="8g",
                    help="docker --memory cap per instance container (e.g. 8g). Bounds a runaway "
                         "search/process so it OOMs the container cleanly (exit 137) instead of the "
                         "host OOM-killer ambiguously SIGKILLing the worker. Empty string disables.")
    ap.add_argument("--model-name", default="ouroboros-e1-pro-gpt-5.5")
    ap.add_argument("--solve-timeout", type=int, default=4500,
                    help="root task timeout for solving /app.")
    ap.add_argument("--no-hard-timeouts", action="store_true",
                    help="do not impose a host subprocess timeout around the task container. "
                         "Use with --solve-timeout 0 for step-budget-style SWE Pro experiments.")
    ap.add_argument("--pretask-evolution", action="store_true",
                    help="OPTIONAL (default OFF): run native task-specific self-evolution before each solve, "
                         "sharing obo-repo/obo-data across the sequence. The baseline path is unchanged "
                         "when this flag is absent.")
    ap.add_argument("--pretask-evolution-wait-max", type=int, default=0,
                    help="max seconds to wait for pre-task evolution absorption; 0 means no hard cutoff")
    ap.add_argument("--solve-step-budget", type=int, default=0,
                    help="optional descriptive solve step/round budget passed to the pre-task evolution prompt")
    ap.add_argument("--cadence", default="every_n:1",
                    help="native post-task evolution cadence: every_n:<k> | llm | off (default every_n:1).")
    ap.add_argument("--review-slots", type=int, default=3,
                    help="reviewer slot count (all pinned to --solve-model). 1 = single reviewer "
                         "(loud single_reviewer_no_diversity). Default 3 (back-compat).")
    ap.add_argument("--review-effort", default="",
                    help="reasoning effort for review + scope-review; empty = take from the profile.")
    ap.add_argument("--runtime-mode", default="",
                    help="OUROBOROS_RUNTIME_MODE override (light|advanced|pro); empty = take from the profile.")
    ap.add_argument("--image-input-mode", default="",
                    help="OUROBOROS_IMAGE_INPUT_MODE override (auto|inline|caption|off); empty = take from the profile.")
    ap.add_argument("--memory-mode", default="",
                    help="per-task solve memory mode (shared|forked|empty); empty = entrypoint default "
                         "(empty child drive, v6.56.0 — the measured artifact is the harness, not carried memory).")
    ap.add_argument("--disable-tools",
                    default="web_search,browse_page,browser_action,analyze_screenshot,vlm_query,view_image,claude_code_edit",
                    help="comma-separated tools withheld from the solve task. Default disables web/browser/vision "
                         "and claude_code_edit. Drop view_image from the list to allow native inline vision.")
    ap.add_argument("--absorb-max", type=int, default=1800,
                    help="max wait for absorbed evolution cycle after a task (seconds). Cycle = "
                         "separate evolution task (review triad) plus os.execvpe restart.")
    ap.add_argument("--reflect-min", type=int, default=30, help="deprecated: wait-until-quiet was replaced by wait-for-absorb")
    ap.add_argument("--reflect-max", type=int, default=900, help="deprecated: see --absorb-max")
    ap.add_argument("--quiet-stable", type=int, default=25, help="deprecated")
    ap.add_argument("--baseline", action="store_true",
                    help="baseline E1': disable the code-evolution channel (POST_TASK_EVOLUTION=false). This is the default; kept for compatibility.")
    ap.add_argument("--evolution", action="store_true",
                    help="enable the native post-task evolution channel for E1v2 comparisons")
    ap.add_argument("--self-improve", action="store_true",
                    help="deprecated alias for --evolution; kept for auto_run compatibility")
    ap.add_argument("--selfimprove-timeout", type=int, default=900,
                    help="deprecated in single-root mode; kept for compatibility")
    ap.add_argument("--reset-state", action="store_true", help="recreate obo-repo/obo-data volumes (clean X0)")
    ap.add_argument("--volume-suffix", default="",
                    help="suffix for obo-repo/obo-data volumes AND container names, e.g. -w1, so parallel "
                         "workers stay isolated (obo-repo-w1/obo-data-w1). Empty = shared default volumes.")
    ap.add_argument("--pause-on-api-err", type=int, default=0,
                    help="pause after a task whose api_errors count exceeds N (manual check: transient interruption vs legitimate recovery). -1 disables pausing")
    args = ap.parse_args()
    args.self_improve = bool(args.evolution or args.self_improve) and not args.baseline

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key and not os.environ.get("OPENAI_API_KEY", "").strip():
        print("error: neither OPENROUTER_API_KEY nor OPENAI_API_KEY is set", file=sys.stderr); return 2

    ensure_util_image()  # pull the --pull=never utility image once, up front

    out_dir = ensure_outside_repo(pathlib.Path(args.out_dir).expanduser(), SRC)
    order = read_full_order() if args.full_set else read_csv_order(pathlib.Path(args.csv).expanduser())
    ids = order[args.start - 1: args.start - 1 + args.limit]
    print(f"[pro] sequence ({len(ids)}): " + " -> ".join(norm(i)[:40] for i in ids), file=sys.stderr)
    rows = load_pro_rows(ids)
    missing = [i for i in ids if i not in rows]
    if missing:
        print(f"[pro] !! missing from dataset (skip): {missing}", file=sys.stderr)

    vsuf = (getattr(args, "volume_suffix", "") or "")
    VREPO, VDATA = "obo-repo" + vsuf, "obo-data" + vsuf
    if args.reset_state:
        for v in (VREPO, VDATA):
            subprocess.run(["docker", "volume", "rm", "-f", v], capture_output=True)
    for v in (VREPO, VDATA):
        subprocess.run(["docker", "volume", "create", v], capture_output=True)

    def atomic_write(p: pathlib.Path, text: str) -> None:
        tmp = p.with_suffix(p.suffix + ".tmp"); tmp.write_text(text, encoding="utf-8"); os.replace(tmp, p)

    preds, timeline = [], []

    def persist() -> None:
        atomic_write(out_dir / "timeline.jsonl", "\n".join(json.dumps(t, ensure_ascii=False) for t in timeline) + "\n")
        atomic_write(out_dir / "predictions.jsonl", "\n".join(json.dumps(p, ensure_ascii=False) for p in preds) + ("\n" if preds else ""))

    for i, cid in enumerate([c for c in ids if c in rows], 1):
        row = rows[cid]
        img = f"{IMG_REPO}:{row['dockerhub_tag']}"
        cid_dir = out_dir / cid.replace("/", "_")
        # Resume: a prior (possibly teardown-killed) invocation already captured a
        # patch for this task. Reconstruct the record from disk with NO Docker calls
        # (no image pull, no state read) and continue. Skipped under --reset-state,
        # which wants a clean fresh solve.
        rr = None if args.reset_state else resume_result(cid, cid_dir, args.model_name)
        if rr is not None:
            res = normalize_result(rr, cid, args)
            if res["model_patch"].strip():
                preds.append({k: res[k] for k in ("instance_id", "model_name_or_path", "model_patch")})
            # spent_after is unknown on resume; recording 0.0 avoids a docker state read.
            timeline.append(build_timeline_row(i, cid, res, 0.0, ["RESUME"]))
            persist()
            print(f"[pro] RESUME task {i}/{len(ids)}: {norm(cid)[:50]} patch.diff exists "
                  f"({len(res['model_patch'])}B), skipped re-solve (no docker)", file=sys.stderr)
            continue
        docker_pull_if_missing(img)
        spent = read_spent_usd(VDATA) if i > 1 else 0.0
        if spent >= args.total_budget:
            print(f"[pro] STOP: budget ${args.total_budget} exhausted (spent ${spent:.2f})", file=sys.stderr); break
        task_total = min(args.total_budget, spent + args.per_task_cost)
        seed = derive_run_settings(args.settings, out_dir, args.solve_model, task_total, args.per_task_cost,
                                   post_task_evolution=args.self_improve, cadence=args.cadence,
                                   review_slots=args.review_slots, review_effort=args.review_effort,
                                   runtime_mode=args.runtime_mode, image_input_mode=args.image_input_mode)
        print(f"\n[pro] === task {i}/{len(ids)}: {norm(cid)[:50]} === spent=${spent:.2f} cap=${task_total:.2f} lang={row.get('repo_language')}", file=sys.stderr)
        res = normalize_result(run_instance(cid, row, args, api_key, seed, task_total), cid, args)
        if res["model_patch"].strip():
            preds.append({k: res[k] for k in ("instance_id", "model_name_or_path", "model_patch")})
        flags = [f for f, on in (("TIMEOUT", res["timed_out"]), ("INFRA", res["infra_suspect"]),
                                 ("ROLLBACK", res["health_rollback"])) if on]
        # EARLY persist BEFORE the teardown. The patch is already captured inside
        # run_instance (read from /out/patch.diff before the container exits). The
        # teardown below — dump_state, then the NEXT task's image pull — can hang for
        # hours on a loaded docker daemon (colima). If the orchestrator kills a
        # teardown-hung run, this record is already on disk, so auto_run sees a LEGIT
        # task (timeline row exists) instead of a phantom failure it re-pulls and
        # re-solves. The post-teardown write below corrects spent_after.
        timeline.append(build_timeline_row(i, cid, res, spent, flags))   # provisional spend
        persist()
        dump_state(cid_dir, VREPO, VDATA, vsuf)
        spent_after = read_spent_usd(VDATA)
        timeline[-1] = build_timeline_row(i, cid, res, spent_after, flags)   # accurate spend
        se = res.get("selfedit") or {}
        print(f"[pro] {norm(cid)[:50]}: patch={len(res['model_patch'])}B spent=${spent_after:.2f} api_err={res['api_errors']} ctx_err={res['api_ctx']} {' '.join(flags) or 'ok'}", file=sys.stderr)
        if args.pretask_evolution:
            pe = res.get("pretask_evolution") or {}
            print(f"[pro]    pretask-evolution: absorbed={pe.get('absorbed', False)} cycles={pe.get('cycles_after', 0)} "
                  f"reason={pe.get('reason', '')} degraded={pe.get('degraded', False)} elapsed={pe.get('elapsed_sec', '?')}", file=sys.stderr)
        if args.self_improve:
            print(f"[pro]    self-edit: commits={se.get('commits_added',0)} loc=+{se.get('loc_added',0)}/-{se.get('loc_removed',0)} "
                  f"tools={len(se.get('tools_added',[]))} verdicts={se.get('verdicts',{})} rollback={se.get('health_rollback',False)}", file=sys.stderr)
        for key in ("solve_line", "refl_line", "quiet_line"):
            if res[key]:
                print(f"[pro]    {res[key]}", file=sys.stderr)
        print(f"[pro]    dump: data={'OK' if (cid_dir/'obo-data.tgz').exists() else 'NO'} repo={'OK' if (cid_dir/'obo-repo.tgz').exists() else 'NO'}", file=sys.stderr)
        persist()
        if args.pause_on_api_err >= 0 and res["api_errors"] > args.pause_on_api_err:
            print(f"\n[pro] ⏸ PAUSED_API_ERR: task {i} ({norm(cid)[:46]}) api_errors={res['api_errors']} > {args.pause_on_api_err}, "
                  f"patch={len(res['model_patch'])}B", file=sys.stderr)
            print("[pro]    MANUAL CHECK: legitimate recovery (real patch, events appended) or transient interruption (0B/few edits).", file=sys.stderr)
            print(f"[pro]    post-task dump saved in {cid.replace('/','_')}/. Rerun this task by restoring volumes to the previous dump and using --start {args.start + i - 1}.", file=sys.stderr)
            break

    print(f"\n[pro] done. tasks={len(timeline)} predictions={len(preds)} -> {out_dir/'predictions.jsonl'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

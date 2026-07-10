#!/usr/bin/env python3
"""Autonomous SWE-Pro range runner with retry-on-network-transient behavior.

Runs run_pro.py one task at a time. After each task:
  - LEGIT (patch exists, OR a demonstrably-executed genuine 0-byte failure): snapshot
    last-good volumes and continue.
  - TRANSIENT (the task demonstrably never ran: empty patch AND (>=3 network api
    errors, or fewer than 2 solve events)): restore last-good volumes, sleep
    --retry-wait, retry the same task.
Transient means the LLM/provider channel failed to sustain the agent run; see the
network-transient retry policy. A genuine 0-byte failure (agent ran, produced no
patch) is recorded as LEGIT and never retried — retrying it would silently turn
pass@1 into conditional best-of-N.
last-good at start is the current volume state (= post-(start-1)).

Parallel shards: pass a unique --volume-suffix per shard (isolates obo-repo/obo-data
volumes, container names, and container cleanup to this shard). Completed indices
leave a sentinel in <out-dir>/done_idx/ so a restarted shard never re-solves an
already-recorded task (k=1 integrity). Every attempt's timeline row is appended to
<out-dir>/timeline_all.jsonl before run_pro rewrites timeline.jsonl, and a retried
attempt's artifacts are archived to <instance>/attempt_N/ — retry disclosure needs
the failed attempts themselves, not just their count.

  OPENROUTER_API_KEY=<fallback .env> python3 pro/auto_run.py --start 27 --end 50 \
      --out-dir runs/pro_e1_27_50 --volume-suffix -w01 --full-set
"""
from __future__ import annotations
import argparse, json, os, pathlib, shutil, subprocess, sys, time
from datetime import datetime, timezone

if __package__ in {None, ""}:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))

from devtools.benchmarks.common.run_roots import ensure_outside_repo
from ouroboros.platform_layer import kill_process_tree, subprocess_new_group_kwargs

HARN = pathlib.Path(__file__).resolve().parent
REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
RUN_PRO = HARN / "run_pro.py"

PERMANENT_INFRA = {"pyexpat_abi_mismatch", "server_import_failed", "pip_bootstrap_failed", "libc_skip"}


def log(msg: str) -> None:
    t = time.strftime("%m-%d %H:%M:%S")
    print(f"[auto {t}] {msg}", file=sys.stderr, flush=True)


def snapshot(dst: pathlib.Path, vsuf: str) -> bool:
    """Dump live obo-data/obo-repo volumes into dst/*.tgz (last-good rollback point).
    Returns True only when BOTH volumes were captured — a caller that starts the run
    (or records a last-good) on a partial snapshot would roll back to blank state."""
    dst.mkdir(parents=True, exist_ok=True)
    ok = True
    for vol, name in ((f"obo-data{vsuf}", "obo-data.tgz"), (f"obo-repo{vsuf}", "obo-repo.tgz")):
        tmp = dst / (name + ".partial")
        r = subprocess.run(["docker", "run", "--rm", "--pull=never", "-v", f"{vol}:/src:ro", "-v", f"{dst}:/dump",
                            "--entrypoint", "tar", "alpine:3", "czf", f"/dump/{name}.partial", "-C", "/src", "."],
                           capture_output=True, timeout=1800)
        if r.returncode == 0 and tmp.exists():
            os.replace(tmp, dst / name)
        else:
            tmp.unlink(missing_ok=True)
            log(f"!! snapshot {name} FAILED rc={r.returncode}")
            ok = False
    return ok


def restore(src: pathlib.Path, vsuf: str) -> bool:
    """Restore obo-data/obo-repo volumes from src/*.tgz. Returns True on full
    success. A missing snapshot tgz or a failed extract must NOT silently leave an
    empty recreated volume (the retry would then run on blank state): skip the
    volume when its snapshot is absent, and log LOUD + return False when an extract
    fails so the caller can treat the retry state as unreliable rather than clean."""
    ok = True
    for vol, name in ((f"obo-data{vsuf}", "obo-data.tgz"), (f"obo-repo{vsuf}", "obo-repo.tgz")):
        tgz = src / name
        if not tgz.is_file():
            log(f"restore SKIP {name}: no snapshot at {tgz} — leaving existing {vol} untouched")
            ok = False
            continue
        subprocess.run(["docker", "volume", "rm", "-f", vol], capture_output=True)
        subprocess.run(["docker", "volume", "create", vol], capture_output=True)
        r = subprocess.run(["docker", "run", "--rm", "--pull=never", "-v", f"{vol}:/d", "-v", f"{src}:/src:ro",
                            "alpine:3", "tar", "xzf", f"/src/{name}", "-C", "/d"], capture_output=True, timeout=1800)
        if r.returncode != 0:
            log(f"restore FAILED {name} rc={r.returncode}: {vol} may be empty — retry state is unreliable")
            ok = False
    return ok


def reflections(vsuf: str) -> int:
    r = subprocess.run(["docker", "run", "--rm", "--pull=never", "-v", f"obo-data{vsuf}:/d:ro", "alpine:3",
                        "sh", "-c", "wc -l </d/logs/task_reflections.jsonl 2>/dev/null || echo 0"],
                       capture_output=True, text=True)
    try:
        return int((r.stdout or "0").strip().split()[0])
    except Exception:
        return -1


def _rm_obopro_containers(vsuf: str) -> None:
    """Remove leftover benchmark containers OF THIS SHARD ONLY (named
    ``obopro<vsuf>-*`` solve and ``obopro<vsuf>-dump-*`` teardown containers) after a
    task-wall-timeout. A bare ``obopro-`` filter would kill every other shard's
    in-flight solve and silently record up to N-1 poisoned 0-byte results. Avoids the
    GNU-only ``xargs -r`` (BSD/macOS xargs lacks it) by listing ids in Python."""
    try:
        ids = subprocess.run(["docker", "ps", "-aq", "--filter", f"name=obopro{vsuf}-"],
                             capture_output=True, text=True, timeout=60).stdout.split()
    except Exception:
        ids = []
    if ids:
        try:
            subprocess.run(["docker", "rm", "-f", *ids], capture_output=True, timeout=300)
        except Exception:
            pass


def _rmi(ref: str) -> None:
    """Remove one completed task's image. Plain (non-forced) rmi: if anything still
    references it (another shard mid-use of a duplicate tag), docker refuses and the
    image survives — losing a few GB of disk beats un-tagging an image in use."""
    try:
        subprocess.run(["docker", "rmi", ref], capture_output=True, timeout=300)
    except Exception:
        pass


def _append_timeline_all(out_dir: pathlib.Path, row: dict, idx: int, attempt: int) -> None:
    rec = dict(row)
    rec["idx"] = idx
    rec["attempt"] = attempt
    rec["logged_at"] = datetime.now(timezone.utc).isoformat()
    try:
        with (out_dir / "timeline_all.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        log(f"!! timeline_all append failed: {e}")


def run_one(i: int, out_dir: pathlib.Path, args, attempt: int) -> dict:
    """Run run_pro once for 1-based task index `i`.

    Returns {"pb": patch_bytes|None, "ae": api_errors|None, "iid": instance_id,
    "degraded": bool, "permanent_skip": bool, "row": timeline_row|None}.
    pb None => run_pro failure / non-executed task (caller retries)."""
    cmd = [sys.executable, str(RUN_PRO), "--start", str(i), "--limit", "1",
           "--out-dir", str(out_dir), "--total-budget", str(args.total_budget),
           "--per-task-cost", str(args.per_task_cost), "--pause-on-api-err", "-1"]
    if args.volume_suffix:
        # `=` form: the suffix conventionally starts with a dash (-w01) and argparse
        # would otherwise parse it as an option flag.
        cmd += [f"--volume-suffix={args.volume_suffix}"]
    if args.full_set:
        cmd += ["--full-set"]
    if args.csv:
        cmd += ["--csv", args.csv]
    if args.settings:
        cmd += ["--settings", args.settings]
    if args.solve_model:
        cmd += ["--solve-model", args.solve_model]
    if args.model_name:
        cmd += ["--model-name", args.model_name]
    if args.review_slots is not None:
        cmd += ["--review-slots", str(args.review_slots)]
    if args.review_effort:
        cmd += ["--review-effort", args.review_effort]
    if args.solve_timeout is not None:
        cmd += ["--solve-timeout", str(args.solve_timeout)]
    if args.memory_mode:
        cmd += ["--memory-mode", args.memory_mode]
    if args.baseline:
        cmd += ["--baseline"]
    env = dict(os.environ)
    for p in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        env.pop(p, None)
    tl = out_dir / "timeline.jsonl"
    tl.unlink(missing_ok=True)        # Freshness: run_pro rewrites timeline; if it did not write (failure/disk-full),
                                      # there is nothing to read -> None -> retry, not a stale previous-task record
    # Launch run_pro in its OWN process group/session (cross-platform via
    # platform_layer) so a wall-clock overrun can kill the whole subprocess tree
    # (run_pro + its docker client + zstd children), not just the direct child —
    # otherwise a hung teardown keeps contending with the next task.
    proc = subprocess.Popen(cmd, env=env, **subprocess_new_group_kwargs())
    try:
        proc.wait(timeout=args.task_wall_timeout)
    except subprocess.TimeoutExpired:
        # Almost always a post-solve teardown stall (volume dump / next image
        # load), NOT the solve itself. Kill the whole process tree (cross-platform),
        # reap the direct child so it does not linger as a zombie, then remove any
        # leftover container OF THIS SHARD. The patch (if any) is already on
        # disk and run_pro writes the timeline row BEFORE teardown, so the read below
        # still sees a LEGIT task instead of a phantom failure that gets re-solved.
        log(f"idx{i} TASK-WALL-TIMEOUT after {args.task_wall_timeout}s — killing run_pro process tree + shard containers; continuing")
        kill_process_tree(proc)
        try:
            proc.wait(timeout=10)
        except Exception:
            pass
        _rm_obopro_containers(args.volume_suffix)
    try:
        rows = [json.loads(l) for l in tl.read_text().splitlines() if l.strip()]
        last = rows[-1]
        _append_timeline_all(out_dir, last, i, attempt)
        if last.get("secret_opt_in_required"):
            # Hard configuration error, NOT a transient: run_pro refused to inject the
            # provider key (OUROBOROS_BENCH_ALLOW_CONTAINER_SECRETS unset), so the task
            # never executed. Stop the whole autonomous run rather than retrying a
            # config error or counting an unexecuted task as LEGIT.
            log("FATAL: OPENROUTER_API_KEY was not injected into the task container "
                "(set OUROBOROS_BENCH_ALLOW_CONTAINER_SECRETS=1 for audited local smoke). Stopping.")
            raise SystemExit(2)
        base = {"iid": last.get("instance_id", "?"),
                "degraded": bool(last.get("evolution_degraded", False)), "row": last}
        if last.get("infra_suspect"):
            reason = str(last.get("infra_reason") or "")
            if reason in PERMANENT_INFRA:
                log(f"idx{i} permanent infra_suspect reason={reason}; recording non-run and continuing without retry")
                return {**base, "pb": 0, "ae": 0, "permanent_skip": True}
            # Task did not actually execute (e.g. musl-image env-volume skip, image
            # eviction). Never snapshot a non-run task as a LEGIT last-good: surface
            # as pb=None so the caller treats it as a failure (retry/stop).
            return {**base, "pb": None, "ae": None, "permanent_skip": False}
        return {**base, "pb": int(last.get("patch_bytes", 0)),
                "ae": int(last.get("api_errors", 0)), "permanent_skip": False}
    except Exception as e:
        log(f"!! timeline was not written after idx{i} (run_pro failure): {e}")
        return {"pb": None, "ae": None, "iid": "?", "degraded": False,
                "permanent_skip": False, "row": None}


def _archive_attempt(out_dir: pathlib.Path, iid: str, attempt: int) -> None:
    """Move a failed attempt's per-instance artifacts to <instance>/attempt_N/ so the
    retry cannot overwrite them (they ARE the retry disclosure). Per-file moves: one
    failure must not abort the rest. patch.diff is force-cleared at the end even if
    its move failed — a stale patch.diff would make run_pro's resume path RESUME the
    failed attempt instead of retrying it."""
    if not iid or iid == "?":
        return
    cdir = out_dir / str(iid).replace("/", "_")
    if not cdir.is_dir():
        return
    adir = cdir / f"attempt_{attempt}"
    try:
        adir.mkdir(exist_ok=True)
    except Exception as e:
        log(f"!! attempt archive mkdir failed for {iid}: {e}")
        adir = None
    if adir is not None:
        for p in list(cdir.iterdir()):
            if p.name.startswith("attempt_"):
                continue
            try:
                shutil.move(str(p), str(adir / p.name))
            except Exception as e:
                log(f"!! attempt archive move failed for {iid}/{p.name}: {e}")
    try:
        (cdir / "patch.diff").unlink(missing_ok=True)
    except Exception as e:
        log(f"!! could not clear stale patch.diff for {iid}: {e}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True, help="inclusive")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--retry-wait", type=int, default=300, help="sleep before retrying a transient (s)")
    ap.add_argument("--max-retries", type=int, default=5, help="max retries for one task before stopping")
    ap.add_argument("--total-budget", type=float, default=500.0)
    ap.add_argument("--per-task-cost", type=float, default=50.0)
    ap.add_argument("--keep-images", type=int, default=2,
                    help="keep the N most recent completed OWN-task images loaded; older own images "
                         "are rmi'd (per-shard bookkeeping — a global newest-N window is ordered by "
                         "image BUILD date and races other shards)")
    ap.add_argument("--task-wall-timeout", type=int, default=9000,
                    help="kill run_pro + its obopro container and continue if one task exceeds this wall-clock "
                         "(s). The captured patch.diff is already on disk and run_pro writes the timeline row "
                         "before teardown, so a teardown stall under a loaded docker daemon does not hang the run.")
    ap.add_argument("--volume-suffix", default="", help="unique per parallel shard, e.g. -w01")
    ap.add_argument("--full-set", action="store_true", help="forward to run_pro (HF-split order)")
    ap.add_argument("--csv", default="", help="forward a custom task-order CSV to run_pro")
    ap.add_argument("--settings", default="", help="forward the settings profile to run_pro")
    ap.add_argument("--solve-model", default="", help="forward to run_pro")
    ap.add_argument("--model-name", default="", help="forward to run_pro")
    ap.add_argument("--review-slots", type=int, default=None, help="forward to run_pro")
    ap.add_argument("--review-effort", default="", help="forward to run_pro")
    ap.add_argument("--solve-timeout", type=int, default=None, help="forward to run_pro")
    ap.add_argument("--memory-mode", default="", help="forward to run_pro")
    ap.add_argument("--baseline", action="store_true", help="forward to run_pro (evolution off)")
    args = ap.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY", "").strip():
        log("error: OPENROUTER_API_KEY is not set"); return 2

    # The utility image backs every snapshot/restore (below) and run_pro's state
    # reads, all with --pull=never. Guarantee it up front — the first snapshot()
    # runs before any run_pro call, so run_pro's own preflight is too late here.
    from devtools.benchmarks.swe_bench_pro.e1v2.run_pro import ensure_util_image
    ensure_util_image()

    vsuf = args.volume_suffix
    out_dir = ensure_outside_repo(pathlib.Path(args.out_dir).expanduser(), REPO_ROOT)
    lastgood = out_dir / "_lastgood"
    done_dir = out_dir / "done_idx"
    done_dir.mkdir(parents=True, exist_ok=True)

    log(f"START autonomous run idx{args.start}..{args.end} vsuf='{vsuf}'; current volume reflections={reflections(vsuf)}")
    log("capturing baseline last-good (= state before first task)...")
    if not snapshot(lastgood, vsuf):
        log("FATAL: baseline last-good snapshot incomplete — refusing to start without a "
            "valid rollback point (a transient would roll back to blank state). Check docker/volumes.")
        return 3

    results = []
    own_images: list[str] = []   # image refs of completed own tasks, oldest first
    cleaned: set[str] = set()
    consecutive_exhausted = 0
    for i in range(args.start, args.end + 1):
        marker = done_dir / f"{i}.json"
        if marker.exists():
            try:
                rec = json.loads(marker.read_text(encoding="utf-8"))
            except Exception:
                rec = {"idx": i, "instance_id": "?", "patch_bytes": -1}
            results.append(rec)
            log(f"idx{i} SKIP-DONE (sentinel exists; not re-solving) :: {str(rec.get('instance_id'))[:46]}")
            continue
        tries = 0
        exhausted = False
        while True:
            r = run_one(i, out_dir, args, attempt=tries)
            pb, ae, row = r["pb"], r["ae"], (r.get("row") or {})
            # Genuine 0-byte failure = the agent demonstrably ran (>=2 solve events),
            # the provider channel was healthy enough (<3 network transients), and the
            # run was not cut short by infra (OOM kill / host timeout leave an empty
            # patch that says nothing about the agent). Anything weaker is a non-run
            # and gets retried; anything recorded here is final (k=1).
            infra_cut = bool(row.get("oom")) or ("TIMEOUT" in (row.get("flags") or []))
            genuine_0b = (pb == 0 and not r["permanent_skip"]
                          and (ae is not None and ae < 3)
                          and int(row.get("n_events", 0)) >= 2
                          and not infra_cut)
            ok = (pb is not None) and (pb > 0 or genuine_0b or r["permanent_skip"])
            if not ok:
                kind = "run_pro-failure" if pb is None else \
                    f"TRANSIENT(0B,api_err={ae},n_events={row.get('n_events', '?')},infra_cut={infra_cut})"
                # Archive under the attempt number that just ran (matches the
                # `attempt` field written to timeline_all.jsonl), THEN bump.
                _archive_attempt(out_dir, r["iid"], tries)
                if not restore(lastgood, vsuf):
                    # Baseline tgz is guaranteed present (start-of-run gate), so a
                    # False here is a failed extract (docker transient). Retry once;
                    # if it still fails, the volume state is unreliable and any
                    # further attempt would corrupt k=1 provenance — stop the shard
                    # (append-only + done_idx make it resumable) rather than proceed.
                    time.sleep(30)
                    if not restore(lastgood, vsuf):
                        log(f"FATAL: restore of last-good failed twice at idx{i} — volume state "
                            f"unreliable; stopping shard to preserve provenance (resume via done_idx).")
                        return 4
                tries += 1
                if tries > args.max_retries:
                    # Retries exhausted on a task that keeps failing to execute
                    # (e.g. deterministic OOM). Record THIS failed attempt as the
                    # final result with full disclosure and move on — one
                    # pathological instance must not stop the whole shard.
                    exhausted = True
                    log(f"idx{i} EXHAUSTED_RETRIES ({args.max_retries}): recording final "
                        f"failure (patch=0B) and continuing :: {str(r['iid'])[:46]}")
                else:
                    log(f"idx{i} {kind} - retry {tries}/{args.max_retries} after "
                        f"{args.retry_wait}s; attempt archived + restore last-good")
                    time.sleep(args.retry_wait)
                    continue
            if ok and r["permanent_skip"]:
                if not restore(lastgood, vsuf):
                    log(f"!! restore of last-good failed after permanent-skip idx{i}; next task starts "
                        f"from current (un-rolled-back) volume state — treat downstream results with care.")
                log(f"idx{i} SKIP_PERMANENT_INFRA: patch={pb}B api_err={ae} degraded={r['degraded']} :: {str(r['iid'])[:46]}")
            elif ok:
                snapshot(lastgood, vsuf)  # new last-good = post-idx_i
                tag = "LEGIT" if pb > 0 else "LEGIT-0B(genuine)"
                log(f"idx{i} {tag}: patch={pb}B api_err={ae} n_events={row.get('n_events', '?')} "
                    f"refl={reflections(vsuf)} degraded={r['degraded']} :: {str(r['iid'])[:46]}")
            # exhausted path: volumes already restored above
            rec = {"idx": i, "instance_id": r["iid"], "patch_bytes": pb if pb is not None else 0,
                   "api_err": ae, "retries": min(tries, args.max_retries),
                   "evolution_degraded": r["degraded"],
                   "permanent_skip": r["permanent_skip"], "genuine_0b": genuine_0b,
                   "exhausted_retries": exhausted, "infra_cut": infra_cut}
            results.append(rec)
            try:
                marker.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
            except Exception as e:
                log(f"!! done-sentinel write failed for idx{i}: {e}")
            ref = str(row.get("image_ref") or "")
            if ref and ref not in own_images:
                own_images.append(ref)
            keep = max(1, args.keep_images)
            for old in own_images[:-keep]:
                if old not in cleaned:
                    _rmi(old)
                    cleaned.add(old)
            if r["degraded"]:
                log(f"idx{i}: evolution degraded (benign telemetry); run continues")
            consecutive_exhausted = consecutive_exhausted + 1 if exhausted else 0
            if consecutive_exhausted >= 2:
                log("two consecutive tasks exhausted retries — provider/infra is likely down; stopping shard.")
                _write_summary(out_dir, results, stopped_at=i)
                return 1
            break

    _write_summary(out_dir, results, stopped_at=None)
    log(f"DONE idx{args.start}..{args.end}: {len(results)} tasks, volume reflections={reflections(vsuf)}")
    return 0


def _write_summary(out_dir: pathlib.Path, results: list, stopped_at) -> None:
    s = {"completed": results, "stopped_at": stopped_at,
         "n_done": len(results), "n_with_patch": sum(1 for r in results if r.get("patch_bytes", 0) > 0),
         "n_genuine_0b": sum(1 for r in results if r.get("genuine_0b")),
         "n_permanent_skip": sum(1 for r in results if r.get("permanent_skip")),
         "n_exhausted_retries": sum(1 for r in results if r.get("exhausted_retries")),
         "total_retries": sum(r.get("retries", 0) for r in results)}
    (out_dir / "auto_summary.json").write_text(json.dumps(s, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())

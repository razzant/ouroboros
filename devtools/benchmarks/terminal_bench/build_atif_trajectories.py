"""Generate ATIF ``agent/trajectory.json`` files for an existing TB job directory.

Harbor Hub leaderboard submissions require a trajectory for every passing
trial (``static_validation._check_passing_trial_trajectories``), and the file
must exist BEFORE the first ``harbor upload`` of that trial (uploader skips
trials that already exist server-side). Runs made before the adapter emitted
trajectories in-container can be backfilled with this converter.

Usage:
    python build_atif_trajectories.py --job-dir <.../job/<timestamp>> \
        [--model openai/gpt-5.5] [--agent-version 6.55.0] \
        [--overwrite] [--validate]

``--validate`` requires harbor installed in the running interpreter (use the
bench venv); building does not.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from atif import build_trajectory, write_trajectory  # noqa: E402


def _trial_model_name(trial_dir: Path) -> str | None:
    try:
        config = json.loads((trial_dir / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    kwargs = ((config.get("agent") or {}).get("kwargs")) or {}
    model = kwargs.get("ouroboros_model")
    return model if isinstance(model, str) and model else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", required=True, type=Path)
    parser.add_argument("--model", default=None, help="fallback model name")
    parser.add_argument("--agent-name", default="Ouroboros")
    parser.add_argument("--agent-version", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    job_dir: Path = args.job_dir
    if not job_dir.is_dir():
        parser.error(f"not a directory: {job_dir}")

    validator_cls = None
    if args.validate:
        try:
            from harbor.utils.trajectory_validator import TrajectoryValidator
        except ImportError:
            parser.error("--validate needs harbor in this interpreter (use bench venv)")
        validator_cls = TrajectoryValidator

    built = skipped = failed = invalid = 0
    for trial_dir in sorted(p for p in job_dir.iterdir() if p.is_dir()):
        agent_dir = trial_dir / "agent"
        if not agent_dir.is_dir():
            continue
        out_path = agent_dir / "trajectory.json"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            trajectory = build_trajectory(
                agent_dir,
                agent_name=args.agent_name,
                agent_version=args.agent_version,
                model_name=_trial_model_name(trial_dir) or args.model,
            )
            if validator_cls is not None:
                validator = validator_cls()
                if not validator.validate(trajectory):
                    invalid += 1
                    print(
                        f"INVALID {trial_dir.name}: {validator.errors[:5]}",
                        file=sys.stderr,
                    )
                    continue
            write_trajectory(agent_dir, trajectory)
            built += 1
        except Exception as exc:  # noqa: BLE001 - per-trial isolation, report and continue
            failed += 1
            print(f"FAILED {trial_dir.name}: {exc!r}", file=sys.stderr)

    print(
        f"trajectories built={built} skipped_existing={skipped} "
        f"invalid={invalid} failed={failed}"
    )
    return 1 if (failed or invalid) else 0


if __name__ == "__main__":
    raise SystemExit(main())

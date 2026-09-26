"""Resource envelope for an unchanged, pinned Cowork shell runner.

Only its Docker resolution is injected. All resource flags are added before
container creation, and exact run labels let the launcher clean up its own
containers and networks after cancellation, including anonymous helper runs.
The disk check refuses new creations; the launcher's guard also observes the
reserve during work. Neither check is a filesystem quota against other users.
The shim also keeps host-only eval-attempt evidence in ``eval_admission/`` of the
run root, which no container mounts: the exact task each runner container served
and every refused eval-container creation.
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import shutil
from collections.abc import Mapping

LABEL_KEY = "org.ouroboros.cowork.run"
ADMISSION_DIR_NAME = "eval_admission"
MIN_FREE_BYTES = 200 * 1024**3
CONTAINER_CPUS = "4"
CONTAINER_MEMORY = "16g"
CONTAINER_PIDS = "512"


def prepare_resource_env(
    run_env: Mapping[str, str],
    *,
    run_root: pathlib.Path,
    docker_host: str,
    resource_root: pathlib.Path,
    min_free_bytes: int = MIN_FREE_BYTES,
) -> dict[str, str]:
    """Resolve Docker once and return the official runner's private environment.

    A preexisting BASH_ENV is refused rather than silently replacing unrelated
    shell startup behavior. No host configuration or daemon state is changed.
    The caller creates run_root/resource_root and records these applied values.
    """
    if run_env.get("BASH_ENV"):
        raise ValueError("Cowork resource limits require an unset BASH_ENV")
    if not docker_host:
        raise ValueError("Cowork resource limits require an explicit Docker host")
    if min_free_bytes < 0:
        raise ValueError("min_free_bytes must be nonnegative")
    found = shutil.which("docker", path=run_env.get("PATH", os.defpath))
    if not found:
        raise ValueError("Docker executable was not found in the incoming PATH")
    scripts = pathlib.Path(__file__).resolve().parent
    real_docker = pathlib.Path(found).resolve()
    if real_docker == scripts / "docker_limits.sh":
        raise ValueError("Docker executable resolves to the resource shim itself")
    root = pathlib.Path(run_root).resolve()
    resource = pathlib.Path(resource_root).resolve(strict=True)
    if not resource.is_dir():
        raise ValueError("resource_root must be an existing directory")
    return {
        **run_env,
        "DOCKER_HOST": docker_host,
        "BASH_ENV": str(scripts / "docker_discovery.sh"),
        "COWORK_REAL_DOCKER": str(real_docker),
        "COWORK_DOCKER_SHIM": str(scripts / "docker_limits.sh"),
        "COWORK_RUN_LABEL": hashlib.sha256(str(root).encode("utf-8")).hexdigest(),
        "COWORK_RESOURCE_ROOT": str(resource),
        "COWORK_MIN_FREE_BYTES": str(min_free_bytes),
        "COWORK_STOP_FILE": str(root / "resource_stop"),
        "COWORK_ADMISSION_DIR": str(root / ADMISSION_DIR_NAME),
        "COWORK_CONTAINER_CPUS": CONTAINER_CPUS,
        "COWORK_CONTAINER_MEMORY": CONTAINER_MEMORY,
        "COWORK_CONTAINER_PIDS": CONTAINER_PIDS,
        "TMPDIR": str(resource),
    }

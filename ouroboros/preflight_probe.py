"""The hermetic gate's own plugin in the candidate's test run, and the admission it freezes beside it.

``preflight_runner`` admits workers once per run (``test_parallelism.admission``: the CPU default within
measured memory, minus what other runs hold) and writes the count into the run's scratch root with a
nonce-named pytest plugin that every pass loads with ``-p``: on the parallel pass it records one file per
xdist worker that really started, which the gate reads back to refuse a nominally parallel pass that ran
on one worker; on both passes it reads ``serial("exclusive")`` the gate's way and registers the derived
``exclusive`` marker, so a candidate repository needs no conftest of its own for the pass split. The
probe's results pointer names the pass's own ``results.json`` for the crash classifier. Nothing here is
part of the repository under test.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import uuid
from typing import Optional


# Require real concurrency when the admitted budget permits it. Physical room
# or an explicit parent allocation may admit one worker; that is not proof of
# concurrency. The host freezes its admitted count beside the nonce probe, so
# a later memory sample cannot change the pass's expected evidence.
MIN_PREFLIGHT_WORKERS = 2


def _preflight_admission():
    """CPU-based default, capped by physical room, a parent test allocation and the other runs' slots."""
    from ouroboros.test_parallelism import admission

    return admission(os.cpu_count() or 1)


def _preflight_worker_count() -> int:
    return _preflight_admission().workers


def allocated_worker_count(temp_root: pathlib.Path) -> int:
    try:
        return int((probe_dir(temp_root) / "admitted-workers").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return _preflight_worker_count()


# Forcing the flags and the plugins still only proves what the gate ASKED for.
# The pass must actually start min(2, admitted budget) workers, observed inside
# the run: a tiny plugin the gate writes into its own
# disposable temp root (never into the candidate worktree) records one file per
# worker id. Reading it back is the difference between "the argv said -n" and
# "concurrency actually happened".
#
# Scope of the guarantee: this closes the SILENT-DOWNGRADE window (an inherited
# worker count, a blocked plugin, a conftest swallowing the flags), all of which
# return green today. It is not a defence against a candidate that forges the
# evidence by writing worker-named files into the gate's temp root — nothing
# in-process can be, since the candidate suite runs with the same privileges.
#
# The name is only a STEM: `install_worker_probe` appends a per-run nonce. `-p
# name` resolves through `sys.path`, and `python -m pytest` puts the candidate
# worktree at `sys.path[0]`, AHEAD of the PYTHONPATH entry the gate prepends. A
# repository that happens to contain a top-level module of the fixed name would
# therefore shadow the probe, report zero workers, and hard-block every green
# parallel run as PREFLIGHT_PARALLELISM_LOST. The nonce cannot be predicted by a
# committed file, so the gate's own module always wins.
WORKER_PROBE_MODULE = "ouroboros_preflight_probe"

WORKER_PROBE_SOURCE = '''"""Written by the Ouroboros preflight gate; not part of the repository under test."""
import os
import pathlib

import pytest

_LOG = pathlib.Path(__WORKER_LOG__)
_RESULTS = pathlib.Path(__RESULTS_POINTER__)


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    # The gate's own reading of `serial("exclusive")`, whatever the candidate's conftest knows: the
    # `exclusive` marker splits the two passes (`-m "not exclusive"`, `-m exclusive`).
    for item in items:
        marker = item.get_closest_marker("serial")
        if marker is not None and marker.args and marker.args[0] == "exclusive":
            item.add_marker(pytest.mark.exclusive)


def pytest_configure(config):
    config.addinivalue_line("markers", "exclusive: derived from serial('exclusive') by the gate; runs alone")
    # Runs in the controller too, where the variable is unset — only workers
    # report, so the file count IS the worker count.
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    if not worker:
        # This pass's own controller: where its evidence plugin writes results.json.
        directory = getattr(config.option, "ci_evidence_dir", None)
        try:
            if directory:
                _RESULTS.write_text(str(pathlib.Path(directory).resolve()), encoding="utf-8")
        except OSError:
            pass
        return
    try:
        _LOG.mkdir(parents=True, exist_ok=True)
        (_LOG / worker).write_text(worker, encoding="utf-8")
    except OSError:
        pass
'''


def probe_dir(temp_root: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(temp_root) / "probe"


def _worker_log_dir(temp_root: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(temp_root) / "workers"


def _results_pointer(temp_root: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(temp_root) / "ci-evidence-dir"


def recorded_session(temp_root: pathlib.Path) -> Optional[dict]:
    """This pass's own results.json, as its controller located it; None when absent or unreadable."""
    try:
        directory = _results_pointer(temp_root).read_text(encoding="utf-8")
        recorded = json.loads((pathlib.Path(directory) / "results.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return recorded if isinstance(recorded, dict) else None


def install_worker_probe(temp_root: pathlib.Path, admitted=None) -> str:
    """Write the worker-count plugin and return the nonce module name to load.

    The nonce is what makes the probe candidate-independent: see the note on
    ``WORKER_PROBE_MODULE``. Callers must pass the RETURNED name to `-p`.
    ``admitted`` is the admission the caller reserved (the gate's, with its ledger row); without one the
    probe admits here.
    """
    module = f"{WORKER_PROBE_MODULE}_{uuid.uuid4().hex}"
    probe = probe_dir(temp_root)
    probe.mkdir(parents=True, exist_ok=True)
    admitted = _preflight_admission() if admitted is None else admitted
    (probe / "admitted-workers").write_text(str(admitted.workers), encoding="utf-8")
    # The admission's facts for the run's evidence: what the host's other runs held when the gate admitted.
    (probe / "admission.json").write_text(json.dumps({
        "workers": admitted.workers, "held_by_other_runs": admitted.host.slots,
        "leases": [list(lease) for lease in admitted.host.leases],
        "unleased_roots": [list(root) for root in admitted.host.strangers], "note": admitted.host.note,
    }), encoding="utf-8")
    # One pass fills the template's own fields: a root that spells one (a caller's `--evidence-dir`) is not read as one.
    fields = {"__WORKER_LOG__": repr(str(_worker_log_dir(temp_root))),
              "__RESULTS_POINTER__": repr(str(_results_pointer(temp_root)))}
    (probe / f"{module}.py").write_text(
        re.sub("|".join(fields), lambda match: fields[match[0]], WORKER_PROBE_SOURCE), encoding="utf-8")
    return module


def recorded_admission(temp_root: pathlib.Path) -> Optional[dict]:
    try:
        return json.loads((probe_dir(temp_root) / "admission.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def observed_worker_ids(temp_root: pathlib.Path) -> set:
    """Distinct xdist worker ids that reported during the parallel pass."""
    try:
        return {entry.name for entry in _worker_log_dir(temp_root).iterdir() if entry.is_file()}
    except OSError:
        return set()

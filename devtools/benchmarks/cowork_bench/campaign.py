"""Provider-metered campaign budget shared by every Cowork invocation.

This is an operator spending bound, not a model-price table. The same provider key's
lifetime usage is compared with one durable baseline across smoke and recovery runs.
Other spending on that key counts conservatively; a changed key or reset counter needs
explicit reconciliation. The caller holds the existing cross-platform file lock for
the whole run, so two launchers cannot independently spend the same remaining budget.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import pathlib
import time
import urllib.request
from contextlib import contextmanager
from typing import Iterator

_KEY_USAGE_URL = "https://openrouter.ai/api/v1/key"


class UsageCounterError(ValueError):
    """A rejected numeric observation, retained without making it budget truth."""

    def __init__(self, observed_usage: float, previous_usage: float) -> None:
        self.observed_usage = observed_usage
        super().__init__(f"provider usage counter decreased or is invalid: "
                         f"observed={observed_usage!r}, previous={previous_usage!r}")


class CampaignPersistenceError(RuntimeError):
    """The durable campaign record was not written, so it no longer bounds spending."""


def validate_usage(usage: float, previous_usage: float) -> None:
    if not math.isfinite(usage) or usage < 0 or usage < previous_usage:
        raise UsageCounterError(usage, previous_usage)


def _read_usage_http(api_key: str, url: str, *, timeout: float) -> float:
    """The single HTTP reader; production runs it only in the disposable worker."""
    request = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {api_key}", "Cache-Control": "no-cache"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return float(payload["data"]["usage"])


def _meter_worker() -> int:
    payload = json.load(sys.stdin)
    try:
        usage = _read_usage_http(payload["api_key"], payload["url"], timeout=payload["timeout"])
    except Exception as exc:
        # The key travels only through stdin, never the command or diagnostic text.
        message = str(exc).replace(payload["api_key"], "[redacted]")
        print(json.dumps({"error_type": type(exc).__name__, "error": message}))
        return 1
    print(json.dumps({"usage": usage}))
    return 0


def key_usage(api_key: str, *, timeout: float = 15) -> float:
    """Bound the whole HTTP read, including headers/body/DNS and worker startup."""
    deadline = time.monotonic() + timeout
    command = [sys.executable, "-I", "-S", str(pathlib.Path(__file__).resolve()), "--read-usage"]
    # Preserve the parent's isolated roots, but keep this credential on stdin only.
    env = {name: value for name, value in os.environ.items() if value != api_key}
    payload = json.dumps({"api_key": api_key, "url": _KEY_USAGE_URL, "timeout": timeout})
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True, env=env) as worker:
        try:
            stdout, stderr = worker.communicate(payload, timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            worker.kill()
            worker.communicate()  # Reap before returning; there is no orphan HTTP read.
            raise TimeoutError("meter HTTP read exceeded its wall-clock deadline") from None
        if time.monotonic() > deadline:
            raise TimeoutError("meter HTTP response arrived after its deadline")
        result = json.loads(stdout) if stdout else {}
        if worker.returncode:
            error = result.get("error") or stderr or "no worker diagnostic"
            raise RuntimeError(f"meter HTTP read failed ({result.get('error_type', worker.returncode)}): "
                               + str(error).replace(api_key, "[redacted]"))
    value = float(result["usage"])
    validate_usage(value, 0.0)
    return value


class CampaignBudget:
    """A locked, monotonic account of the owner's campaign spending limit."""

    def __init__(self, path: pathlib.Path, *, fingerprint: str, ceiling: float,
                 usage: float, prior_spend: float = 0.0) -> None:
        self.path = path
        if not all(math.isfinite(x) for x in (ceiling, usage, prior_spend)):
            raise ValueError("campaign amounts must be finite")
        if ceiling <= 0 or usage < 0 or prior_spend < 0:
            raise ValueError("campaign ceiling must be positive; usage cannot be negative")
        if path.exists():
            self.record = json.loads(path.read_text(encoding="utf-8"))
            if self.record["fingerprint"] != fingerprint or self.record["ceiling_usd"] != ceiling:
                raise ValueError("campaign key/ceiling changed; reconcile the existing campaign first")
            if prior_spend:
                raise ValueError("prior spending can only be set when creating a campaign")
            if self.record.get("active_run"):
                raise ValueError("previous campaign run has unsettled custody; inspect its processes and containers")
        else:
            self.record = {
                "schema": "ouroboros.cowork_bench.campaign.v1",
                "fingerprint": fingerprint, "ceiling_usd": ceiling,
                "usage_baseline": usage, "prior_spend_usd": prior_spend,
                "last_usage": usage, "created_at": time.time(), "runs": [],
            }
        self.observe(usage)

    @property
    def spent(self) -> float:
        return self.record["prior_spend_usd"] + self.record["last_usage"] - self.record["usage_baseline"]

    @property
    def remaining(self) -> float:
        return self.record["ceiling_usd"] - self.spent

    def save(self) -> None:
        from devtools.benchmarks.common.manifests import write_json

        self.record.update({"spent_usd": self.spent, "remaining_usd": self.remaining,
                            "observed_at": time.time()})
        try:
            write_json(self.path, self.record)
        except Exception as exc:
            raise CampaignPersistenceError(f"campaign record not saved: {type(exc).__name__}: {exc}") from exc

    def observe(self, usage: float) -> None:
        # Only a finite, nonnegative, nondecreasing value reaches the durable record.
        validate_usage(usage, self.record["last_usage"])
        self.record["last_usage"] = usage
        self.save()

    def start(self, run_root: pathlib.Path) -> None:
        self.record["active_run"] = str(run_root)
        self.save()

    def finish(self, run_root: pathlib.Path, *, outcome: str, meter_error: str = "") -> None:
        self.record["runs"].append({"run_root": str(run_root), "outcome": outcome,
                                    "spent_usd": self.spent, "meter_error": meter_error,
                                    "finished_at": time.time()})
        self.record.pop("active_run", None)
        self.save()


@contextmanager
def campaign_lock(path: pathlib.Path) -> Iterator[None]:
    from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    fd = acquire_exclusive_file_lock(lock_path, timeout_sec=0.2, owner_aware_stale=True)
    if fd is None:
        raise RuntimeError("another launcher owns this campaign budget")
    try:
        yield
    finally:
        release_exclusive_file_lock(lock_path, fd)


if __name__ == "__main__":
    if sys.argv[1:] != ["--read-usage"]:
        raise SystemExit("campaign.py is an internal meter worker")
    raise SystemExit(_meter_worker())

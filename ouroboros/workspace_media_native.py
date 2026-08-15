"""The media EXPORT channel: video frames off a target, and nothing else.

Split out of ``workspace_native`` at the module-size gate (plan §5) rather than
grandfathered, and this is the natural seam: frame extraction is one export
channel with one external helper (ffmpeg), so it neither needs nor should have
access to the rest of the native kernel.

The channel's export-policy check does NOT live here. It happens at PREPARE, in
``workspace_native.prepare_native_operation``, against the source path the model
named — before ffmpeg is resolved, before a subprocess exists, and therefore
before a single byte of an excluded file has been read. Judging the source here
would mean the policy ran after the door was already open.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping
from typing import Any

from ouroboros.export_policy_contract import (
    QUESTION_EXPORT,
    export_disclosure_block,
)
from ouroboros.platform_layer import (
    kill_process_tree,
    process_group_id,
    subprocess_new_group_kwargs,
)
from ouroboros.workspace_diagnostics import ToolExecutionEnvelope
from ouroboros.workspace_native_contract import (
    NativeExecutionControl,
    NativeOperationResult,
)

_FRAME_TIMEOUT_SEC = 60


def ffmpeg_binary(configured: str = "", expected_sha256: str = "") -> str:
    """Resolve the helper absolutely; execd additionally verifies its pinned digest.

    NOT cached, and that is the whole point. It was `@functools.lru_cache(maxsize=4)`,
    so the SHA-256 ran on the first call with a given `(configured, expected_sha256)`
    pair and every later call in that process returned the memo — which means the
    pinned digest verified the bytes that were on disk once, at some earlier moment,
    and never again. Detecting a swap AFTER the first verification is precisely what
    pinning a digest is for, so a cache here does not make the check cheaper, it makes
    it a no-op for the case it exists to catch.

    Caching on `(path, mtime_ns, size)` was the alternative and is rejected: an
    attacker who can write into execd's private stage can also `utime` the file, so a
    key made of metadata restores the same hole behind one more step. The cost is
    bounded and dwarfed by what it gates — one hash of the helper (tens of MB, tens of
    milliseconds) per `extract_video_frames` call, i.e. once per operation, before a
    subprocess that then decodes video.
    """

    candidate = str(configured or "").strip()
    if not candidate:
        try:
            import imageio_ffmpeg

            candidate = str(imageio_ffmpeg.get_ffmpeg_exe() or "")
        except Exception:
            candidate = str(shutil.which("ffmpeg") or "")
    if not candidate:
        raise FileNotFoundError("ffmpeg is unavailable on the target")
    resolved = pathlib.Path(candidate).expanduser().resolve(strict=True)
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise PermissionError("ffmpeg helper is not an executable regular file")
    if configured:
        if (
            len(expected_sha256) != 64
            or hashlib.sha256(resolved.read_bytes()).hexdigest() != expected_sha256
        ):
            raise PermissionError("bundled ffmpeg helper failed integrity verification")
    return str(resolved)


def extract_video_frames(
    root: pathlib.Path,
    source: pathlib.Path,
    args: Mapping[str, Any],
    *,
    control: NativeExecutionControl | None,
    kill_process_group: Callable[[subprocess.Popen[Any]], None],
    native_facts: Mapping[str, Any] | None = None,
) -> NativeOperationResult:
    """Extract up to a dozen frames from an ALREADY-AUTHORIZED source path.

    ``source`` is resolved and policy-checked by the caller: this function does no
    confinement and no policy of its own, which is the only way there is exactly
    one place either decision is made.
    """

    try:
        max_frames = max(1, min(12, int(args.get("max_frames") or 5)))
    except (TypeError, ValueError):
        max_frames = 5
    raw_times = str(args.get("timestamps") or "")
    timestamps = [
        value.strip() for value in re.split(r"[,\s]+", raw_times)
        if value.strip()
    ][:max_frames]
    if not timestamps:
        timestamps = [str(index) for index in range(max_frames)]
    blobs: dict[str, bytes] = {}
    artifacts: list[dict[str, Any]] = []
    errors: list[str] = []
    ffmpeg = ffmpeg_binary(
        os.environ.get("OUROBOROS_EXECD_FFMPEG", ""),
        os.environ.get("OUROBOROS_EXECD_FFMPEG_SHA256", ""),
    )
    for index, stamp in enumerate(timestamps, 1):
        try:
            float(stamp)
        except ValueError:
            errors.append(f"invalid timestamp {stamp!r}")
            continue
        with tempfile.NamedTemporaryFile(suffix=".jpg") as handle:
            command = [
                ffmpeg,
                "-nostdin",
                "-loglevel",
                "error",
                "-ss",
                stamp,
                "-i",
                str(source),
                "-frames:v",
                "1",
                "-y",
                handle.name,
            ]
            proc = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                **subprocess_new_group_kwargs(),
            )
            if (pgid := process_group_id(proc.pid)) <= 0 and control is not None:
                kill_process_tree(proc)
                raise RuntimeError("could not resolve ffmpeg process group")
            registered = False
            try:
                if control is not None:
                    try:
                        control.register_process(pgid=pgid)
                    except Exception:
                        kill_process_group(proc)
                        raise
                    registered = True
                deadline = time.monotonic() + _FRAME_TIMEOUT_SEC
                while proc.poll() is None:
                    if control is not None and control.cancelled():
                        kill_process_group(proc)
                        raise InterruptedError("remote frame extraction cancelled")
                    if time.monotonic() >= deadline:
                        kill_process_group(proc)
                        raise subprocess.TimeoutExpired(command, _FRAME_TIMEOUT_SEC)
                    time.sleep(0.05)
                _stdout, stderr = proc.communicate()
                if proc.returncode:
                    raise RuntimeError(stderr.decode("utf-8", errors="replace"))
                data = pathlib.Path(handle.name).read_bytes()
            finally:
                if registered and control is not None:
                    control.release_process(pgid=pgid)
        digest = hashlib.sha256(data).hexdigest()
        blobs[digest] = data
        artifacts.append(
            {
                "name": f"{source.stem}_frame_{index:02d}.jpg",
                "blob_id": digest,
                "sha256": digest,
                "size": len(data),
                "mime": "image/jpeg",
                "timestamp": stamp,
            }
        )
    if not artifacts:
        detail = "; ".join(errors[:5]) or "no frames produced"
        return NativeOperationResult(
            ToolExecutionEnvelope(
                text=f"⚠️ EXTRACT_VIDEO_FRAMES_UNAVAILABLE: {detail}",
                trace={
                    "completion": "complete",
                    "source": source.relative_to(root).as_posix(),
            # Emitted on EVERY media export, frames or none. This channel used to
            # return no disclosure block at all, and Home's judge treats an absent
            # manifest as "nothing was withheld" — a positive claim. So silence from
            # the one door that never spoke read exactly like a clean export, and
            # nothing on Home could tell the two apart. The SOURCE is declared as
            # exported because that is what the frames are derived from, which gives
            # the returned-manifest check something to re-evaluate.
            **export_disclosure_block(
                native_facts, [], [source.relative_to(root).as_posix()],
                question=QUESTION_EXPORT,
            ),
                },
            )
        )
    warning = f"\nWarnings: {'; '.join(errors[:5])}" if errors else ""
    return NativeOperationResult(
        ToolExecutionEnvelope(
            text=json.dumps({"frames": artifacts}, sort_keys=True) + warning,
            artifacts=tuple(artifacts),
            trace={
                "completion": "complete",
                "source": source.relative_to(root).as_posix(),
        # Emitted on EVERY media export, frames or none. This channel used to
            # return no disclosure block at all, and Home's judge treats an absent
            # manifest as "nothing was withheld" — a positive claim. So silence from
            # the one door that never spoke read exactly like a clean export, and
            # nothing on Home could tell the two apart. The SOURCE is declared as
            # exported because that is what the frames are derived from, which gives
            # the returned-manifest check something to re-evaluate.
            **export_disclosure_block(
                native_facts, [], [source.relative_to(root).as_posix()],
                question=QUESTION_EXPORT,
            ),
            },
        ),
        blobs,
    )

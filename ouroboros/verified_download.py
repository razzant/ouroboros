"""Exact byte verification and atomic cached downloads, independent of consumers.

Consumers own artifact policy, error codes and transport timeouts. This module
only binds the bytes to their declared size/digest and publishes complete files.
"""
from __future__ import annotations

import hashlib
import os
import pathlib
from typing import Any
import uuid


class VerifiedDownloadError(RuntimeError):
    """An exact file was absent, damaged or could not be downloaded."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code


def verify_exact_file(
    path: "str | pathlib.Path",
    *,
    size_bytes: int,
    sha256: str,
    code_prefix: str,
    label: str,
    error_type: type[RuntimeError] = VerifiedDownloadError,
) -> pathlib.Path:
    """Verify one review-bound download without interpreting its contents."""
    archive = pathlib.Path(path)
    try:
        size = archive.stat().st_size
    except OSError as exc:
        raise error_type(
            f"{code_prefix}_missing", f"{label} is unavailable: {type(exc).__name__}: {exc}"
        ) from exc
    if size != size_bytes:
        raise error_type(
            f"{code_prefix}_size_mismatch",
            f"{label} size {size} does not match the reviewed {size_bytes}",
        )
    digest = hashlib.sha256()
    try:
        with archive.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise error_type(
            f"{code_prefix}_unreadable", f"{label} read failed: {type(exc).__name__}: {exc}"
        ) from exc
    actual = digest.hexdigest()
    if actual != sha256:
        raise error_type(
            f"{code_prefix}_digest_mismatch",
            f"{label} sha256 {actual} does not match the reviewed {sha256}",
        )
    return archive



def fetch_exact_file(
    *,
    url: str,
    destination: "str | pathlib.Path",
    verify: Any,
    size_bytes: int,
    overflow_code: str,
    failure_code: str,
    label: str,
    timeout_sec: float,
    error_type: type[RuntimeError] = VerifiedDownloadError,
    progress: "dict | None" = None,
) -> pathlib.Path:
    """Atomically fetch one exact file; an existing verified cache wins. ``progress`` (optional)
    receives the bytes written so far, for a caller reporting a download it does not wait out."""
    # Imported here so ``verify_exact_file`` needs only the standard library (a
    # Docker build stage verifies the pinned Chromium archive with this module alone).
    from ouroboros.utils import replace_atomic

    target = pathlib.Path(destination)
    try:
        return verify(target)
    except error_type:
        pass
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        import httpx

        written = 0
        with httpx.Client(follow_redirects=True, timeout=timeout_sec) as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                with temporary.open("xb") as sink:
                    for chunk in response.iter_bytes():
                        if not chunk:
                            continue
                        written += len(chunk)
                        if progress is not None:
                            progress["bytes"] = written
                        if written > size_bytes:
                            raise error_type(
                                overflow_code, f"{label} download exceeded the reviewed size"
                            )
                        sink.write(chunk)
                    sink.flush()
                    os.fsync(sink.fileno())
        verify(temporary)
        replace_atomic(temporary, target)
        return verify(target)
    except error_type:
        raise
    except Exception as exc:
        raise error_type(
            failure_code, f"{label} download failed: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass

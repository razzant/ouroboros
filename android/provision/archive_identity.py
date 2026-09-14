"""Content identity for Gitiles archives whose container timestamps are regenerated."""
import hashlib
import os
import shutil
import urllib.request
import json
import tarfile


def archive_content_sha256(path):
    records = []
    with tarfile.open(path) as handle:
        for member in handle.getmembers():
            stream = handle.extractfile(member) if member.isfile() else None
            value = hashlib.sha256()
            if stream is not None:
                with stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        value.update(block)
            records.append({"path": member.name, "type": member.type.decode("ascii"),
                            "mode": member.mode & 0o777, "linkname": member.linkname,
                            "sha256": value.hexdigest() if member.isfile() else None})
    records.sort(key=lambda item: item["path"])
    return hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verify_artifact(path, pin):
    try:
        if "archive_content_sha256" in pin:
            return archive_content_sha256(path) == pin["archive_content_sha256"]
        return (path.stat().st_size == pin["size_bytes"] and file_sha256(path) == pin["sha256"])
    except (OSError, ValueError, tarfile.TarError):
        return False


def download(artifact, cache):
    identity = artifact.get("archive_content_sha256", artifact.get("sha256"))
    target = cache / (identity + "-" + artifact["name"])
    if target.is_file() and verify_artifact(target, artifact):
        return target
    partial = target.with_name(target.name + ".partial")
    if partial.is_file() and verify_artifact(partial, artifact):
        os.replace(partial, target)
        return target
    # Gitiles may regenerate container bytes between requests at one source commit.
    offset = partial.stat().st_size if partial.exists() and "archive_content_sha256" not in artifact else 0
    request = urllib.request.Request(artifact["url"], headers={"Range": f"bytes={offset}-"} if offset else {})
    with urllib.request.urlopen(request, timeout=60) as response:
        append = bool(offset and response.status == 206
                      and response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"))
        with partial.open("ab" if append else "wb") as output:
            shutil.copyfileobj(response, output, 1024 * 1024)
    if not verify_artifact(partial, artifact):
        partial.unlink()
        raise RuntimeError("Dependency failed size/SHA-256 verification: " + artifact["name"])
    os.replace(partial, target)
    return target

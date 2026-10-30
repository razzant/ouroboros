"""What the qualification reads from one disposable installation: endpoint, records, Git layers, packages.

Read-only toward the product except the owner actions named in the cases (settings, project markers,
prepare/activate). Git reads use ``GIT_OPTIONAL_LOCKS=0`` and never refresh the index. The records read
here are the product's own (``state/server_port.bindings.json``, ``state/private_copies/*.json``,
``logs/supervisor.jsonl``, ``state/body_adoption/handoff.json``). Their shapes were read against the finished
producers at d7bf045d (round 5, HANDOFF.md) and none differs; none is measured on Windows until the native run.
A record that cannot be read is an unmeasured check, never a pass.
"""
import glob
import hashlib
import json
import os
import pathlib
import subprocess
import time
import urllib.error
import urllib.request

from wq_verdict import same_path


class Install:
    def __init__(self, mode_root, port):
        self.root = pathlib.Path(mode_root)
        self.app_root = self.root / "app root"  # a space: quoting of every command the product builds
        self.repo, self.data = self.app_root / "repo", self.app_root / "data"
        self.user_base = self.data / "state" / "python-userbase"
        self.private = self.data / "state" / "python-private"
        self.witness_dir, self.logs = self.root / "witness", self.root / "logs"
        self.requested_port, self.boot_failed = port, False

    # ------------------------------------------------------------------ endpoint
    def binding(self):
        try:
            value = json.loads((self.data / "state" / "server_port.bindings.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        main = value.get("main") if isinstance(value, dict) else None
        return main if isinstance(main, dict) else None

    def http(self, method, path, body=None, port=None, timeout=30):
        """``(status, json-or-text)``; transport failure is ``(None, error)``, never an answer."""
        port = port or (self.binding() or {}).get("port") or self.requested_port
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, method=method,
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw, status = response.read(), response.status
        except urllib.error.HTTPError as exc:
            raw, status = exc.read(), exc.code
        except (OSError, ValueError) as exc:
            return None, repr(exc)
        try:
            return status, json.loads(raw.decode("utf-8") or "null")
        except ValueError:
            return status, raw.decode("utf-8", "replace")[:2000]

    def server(self, custody):
        """The main binding's process if it is alive, its birth equals the binding's and it is in the Job."""
        binding = self.binding()
        if not binding or not isinstance(binding.get("fingerprint"), dict):
            return None
        row = custody.win.identity(int(binding.get("pid") or 0), custody.job)
        if row.get("alive") and row.get("in_job") and row.get("birth") == str(
                binding["fingerprint"].get("creation_time")):
            return {**row, "port": binding.get("port")}
        return None

    def wait_ready(self, custody, deadline, exclude=()):
        """A new ready generation: a live bound server in the Job (not one of ``exclude``) whose
        ``/api/health`` says ``boot.phase == ready``. ``None`` at the deadline; ``boot_failed`` then tells a
        terminal answer (the last health read of a new server said ``boot.phase == failed``) from an
        unmeasured start (no new server, no answer, or still starting)."""
        end, last = time.monotonic() + deadline, {}
        self.boot_failed = False
        while time.monotonic() < end:
            server = self.server(custody)
            if server and (server["pid"], server["birth"]) not in set(exclude):
                status, health = self.http("GET", "/api/health", port=server["port"], timeout=10)
                last = {"server": server, "health_status": status, "health": health}
                if status == 200 and isinstance(health, dict) and health.get("status") == "ok" \
                        and isinstance(health.get("boot"), dict) and health["boot"].get("phase") == "ready":
                    custody.raw.emit("ready", mode=custody.label, **last)
                    return {**server, "health": health}
            time.sleep(1)
        health = last.get("health") if isinstance(last.get("health"), dict) else {}
        self.boot_failed = isinstance(health.get("boot"), dict) and health["boot"].get("phase") == "failed"
        custody.raw.emit("not_ready", mode=custody.label, deadline=deadline, last=last, boot_failed=self.boot_failed,
                         members=custody.members())
        return None

    def endpoint_gone(self, server):
        """The recorded server process is gone and its port does not answer ``/api/health``."""
        status, _ = self.http("GET", "/api/health", port=server.get("port"), timeout=5)
        return status is None

    # ------------------------------------------------------------------ records
    def candidate(self, candidate_id):
        for path in sorted(glob.glob(str(self.data / "state" / "private_copies" / "*.json"))):
            try:
                row = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(row, dict) and row.get("kind") == "update_candidate" and row.get("candidate_id") == candidate_id:
                return row
        return None

    def supervisor_rows(self):
        rows = []
        for path in sorted(glob.glob(str(self.data / "logs" / "supervisor.jsonl*"))):
            try:
                text = pathlib.Path(path).read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for line in text.splitlines():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
        return rows

    def handoff(self):
        try:
            return json.loads((self.data / "state" / "body_adoption" / "handoff.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def witnesses(self):
        rows = []
        for path in glob.glob(str(self.witness_dir / "*.json")):
            try:
                rows.append(json.loads(pathlib.Path(path).read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
        return rows

    # ------------------------------------------------------------------ owner actions (provider-free)
    def set_channel(self, channel):
        status, body = self.http("POST", "/api/settings", {"OUROBOROS_UPDATE_CHANNEL": channel})
        read_status, settings = self.http("GET", "/api/settings")
        seen = settings.get("OUROBOROS_UPDATE_CHANNEL") if isinstance(settings, dict) else None
        if seen is None and isinstance(settings, dict) and isinstance(settings.get("settings"), dict):
            seen = settings["settings"].get("OUROBOROS_UPDATE_CHANNEL")
        return status in (200, 201) and seen == channel, {"post": status, "body": body, "read": read_status,
                                                            "seen": seen}

    def add_marker(self, marker_id):
        status, body = self.http("POST", "/api/projects", {"name": f"Qualification {marker_id}", "id": marker_id})
        return (None if status is None else status in (200, 201)), {"status": status, "body": body}

    def markers(self, names):
        """Which of the owner's marker ids the serving process lists now; ``None`` when it cannot answer.
        The ids are unique words, looked up in the whole answer (not tied to one field spelling)."""
        status, body = self.http("GET", "/api/projects")
        if status != 200 or not isinstance(body, dict):
            return None
        text = json.dumps(body)
        return {name: f'"{name}"' in text for name in names}


# ---------------------------------------------------------------------- Git, read-only unless named
GIT_READ_ENV = {"GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C"}


def git(repo, *args, env=None, check=True, raw=False):
    done = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, timeout=600,
                          env={**(env or os.environ), **GIT_READ_ENV})
    if check and done.returncode != 0:
        raise RuntimeError(f"git {' '.join(args[:3])}: rc={done.returncode} {done.stderr.decode(errors='replace')[-600:]}")
    return done.stdout if raw else done.stdout.decode("utf-8", "replace").strip()


def layers(repo, sentinels=(), env=None):
    """H, the branch, I (the index entries), W/U (porcelain v2 status incl. untracked) and sentinel bytes."""
    try:
        return {"head": git(repo, "rev-parse", "HEAD", env=env),
                "branch": git(repo, "symbolic-ref", "-q", "HEAD", env=env, check=False),
                "index": hashlib.sha256(git(repo, "ls-files", "--stage", "-z", env=env, raw=True)).hexdigest(),
                "status": hashlib.sha256(git(repo, "--no-optional-locks", "status", "--porcelain=v2", "-z",
                                             "--untracked-files=all", "--ignored=no", env=env, raw=True)).hexdigest(),
                "files": {name: file_digest(pathlib.Path(repo) / name) for name in sentinels}}
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        return {"error": repr(exc)}


def file_digest(path):
    try:
        return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def path_id(path):
    """``[volume serial, NTFS file index]`` of the entry itself (a rename keeps it), ``None`` if absent."""
    try:
        st = os.lstat(path)
    except OSError:
        return None
    return [st.st_dev, st.st_ino]


def inventory(site_packages):
    """Distributions in one site-packages: ``{name: {"version", "id"}}`` (the dist-info directory's identity:
    a reinstall makes a new one, a rename keeps it). ``None`` when the directory is absent."""
    root = pathlib.Path(site_packages)
    if not root.is_dir():
        return None
    rows = {}
    for info in sorted(root.glob("*.dist-info")):
        name, _, version = info.name[:-len(".dist-info")].partition("-")
        rows[name.lower().replace("_", "-")] = {"version": version, "id": path_id(info)}
    return rows


def user_site(user_base):
    """The Windows user site of a 3.10 interpreter under ``user_base``."""
    return pathlib.Path(user_base) / "Python310" / "site-packages"


def venv_site(root):
    return pathlib.Path(root) / "Lib" / "site-packages"


def same_inventory(before, after):
    return before is not None and after is not None and before == after


CONTENT_LIMIT = {"files": 250000, "bytes": 8 * 2 ** 30}


def content_inventory(root, limit=None):
    """``{relative path: [size, sha256]}`` of every file under ``root`` except the interpreter's own bytecode
    cache (``__pycache__`` directories, ``*.pyc``): a package or metadata changed in place shows here even
    when its dist-info directory keeps its identity. ``None`` (unmeasured) when ``root`` is absent, a file
    cannot be read or the tree exceeds ``limit``; links are recorded by their target text, never followed."""
    limit, root = limit or CONTENT_LIMIT, pathlib.Path(root)
    if not root.is_dir():
        return None
    rows, total = {}, 0
    try:
        for directory, dirnames, filenames in os.walk(root, onerror=_raise):
            dirnames[:] = sorted(name for name in dirnames if name != "__pycache__")
            for name in sorted(filenames):
                if name.endswith(".pyc"):
                    continue
                path = os.path.join(directory, name)
                relative = os.path.relpath(path, root).replace(os.sep, "/")
                if os.path.islink(path):
                    rows[relative] = ["link", os.readlink(path)]
                    continue
                digest, size = hashlib.sha256(), 0
                with open(path, "rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        digest.update(chunk)
                        size += len(chunk)
                total += size
                rows[relative] = [size, digest.hexdigest()]
                if len(rows) > limit["files"] or total > limit["bytes"]:
                    return None
    except OSError:
        return None
    return rows


def _raise(error):
    raise error


def content_changes(before, after):
    """Paths added, removed or changed between two content inventories; ``None`` when either is unmeasured."""
    if before is None or after is None:
        return None
    return sorted(path for path in set(before) | set(after) if before.get(path) != after.get(path))


def console_launcher_parts(data):
    """distlib's Windows launcher split as ``body_switch`` reads it: (stub, shebang line, archive)."""
    import struct
    eocd = data.rfind(b"PK\x05\x06", max(0, len(data) - 65557))
    if eocd < 0 or len(data) < eocd + 22:
        return None
    size, offset = struct.unpack("<II", data[eocd + 12:eocd + 20])
    archive = eocd - size - offset
    start = data.rfind(b"#!", 0, archive) if archive > 0 else -1
    if start < 0:
        return None
    end = min([at for at in (data.find(b"\r", start, archive), data.find(b"\n", start, archive)) if at >= 0]
              or [archive])
    return data[:start], data[start:end].decode("utf-8", "replace"), data[end:]


def shebang_names(line, python):
    """A ``#!`` line names exactly ``python`` (quoted when it holds a space)."""
    text = line[2:].strip()
    named = text[1:text.find('"', 1)] if text.startswith('"') else text.split(" ")[0]
    return same_path(named, python) and (" " not in str(python) or text.startswith('"'))

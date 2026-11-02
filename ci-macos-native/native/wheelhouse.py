"""Public wheel folder for the native qualification's venv fixture (test equipment, DISCLOSED SUBSTITUTE for the index).

Why: the venv fixture (qb1.sh venvprep, venv_lab.py) installs the candidate's runtime lock with pip from a local
folder (PIP_NO_INDEX=1, PIP_FIND_LINKS=<folder>), the way the candidate's own install_deps runs pip against the
index. On the lab VM that folder held wheels repacked from an installed app (wheelhouse_lab.py), which no one else can
reproduce. Here every wheel comes from PyPI and is fixed in advance by the candidate's own files:
  - the requirement set is the candidate's requirements-runtime.lock: every line whose marker holds for CPython 3.10
    on macOS arm64 (the venv's base interpreter is the 7.7.0 app's Python or a copy of it); a marker of any other
    shape is refused, never guessed;
  - each line's file is one wheel of exactly that version listed in the candidate's uv.lock (registry pypi.org),
    chosen by pip's own tag preference (packaging.tags: CPython 3.10, macOS arm64), and pinned by uv.lock's sha256 and
    size. The choice must be the same for macOS 11.0, 16.0 and 26.0, or the line is refused. A line without such a
    wheel is refused: nothing is built from source.
Nothing here downloads. derive writes the pins; check derives them again and requires the same bytes; report reads a
staged folder, requires exactly the pinned files with their sizes and digests, and writes the two files qb1.sh reads
(wheelhouse-report.json with lock_sha256, and SHA256SUMS). Standard library only; runs on Python 3.10.

  derive LOCK UVLOCK OUT          write the pins (create-only)
  check LOCK UVLOCK PINS          exit 0 only if derive would write exactly PINS
  report PINS LOCK DIR            DIR holds exactly the pinned wheels: write DIR/wheelhouse-report.json, DIR/SHA256SUMS
"""
import hashlib
import json
import os
import re
import sys

HEADER = "# wheelhouse.py pins v1: one PyPI wheel per applicable lock line (name, version, file, size, sha256, url)"
ENVIRONMENT = {"platform_python_implementation": "CPython", "implementation_name": "cpython", "sys_platform": "darwin",
               "platform_machine": "arm64", "os_name": "posix", "platform_system": "Darwin"}
PYTHON_RANGE = ("3.10.0", "3.10.99")   # every 3.10.x must give the same marker result
MACOS_VERSIONS = ((11, 0), (16, 0), (26, 0))   # the choice must not depend on which of these pip reports
REGISTRY = 'source = { registry = "https://pypi.org/simple" }'
WHEEL_RE = re.compile(r'\{ url = "(https://files\.pythonhosted\.org/[^"]+\.whl)", hash = "sha256:([0-9a-f]{64})", '
                      r'size = ([0-9]+)[,}]')
MARKER_RE = re.compile(r"^(\w+) (==|!=|<=|>=|<|>) '([^']*)'$")


def canon(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def version_tuple(text):
    parts = text.split(".")
    if not all(part.isdigit() for part in parts):
        raise ValueError("not a plain version: %r" % text)
    return tuple(int(part) for part in parts)


def compare(left, op, right):
    return {"==": left == right, "!=": left != right, "<": left < right, "<=": left <= right, ">": left > right,
            ">=": left >= right}[op]


def marker_holds(marker, python):
    """or of and of `name op 'value'`; refused (ValueError) for any other shape or name."""
    if "(" in marker or ")" in marker or '"' in marker:
        raise ValueError("unsupported marker shape: %r" % marker)
    result = False
    for alternative in marker.split(" or "):
        holds = True
        for atom in alternative.split(" and "):
            match = MARKER_RE.match(atom.strip())
            if not match:
                raise ValueError("unsupported marker atom: %r" % atom)
            name, op, value = match.groups()
            if name in ("python_full_version", "python_version"):
                have = version_tuple(python if name == "python_full_version" else ".".join(python.split(".")[:2]))
                holds = holds and compare(have, op, version_tuple(value))
            elif name in ENVIRONMENT:
                if op not in ("==", "!="):
                    raise ValueError("unsupported comparison for %s: %r" % (name, atom))
                holds = holds and compare(ENVIRONMENT[name], op, value)
            else:
                raise ValueError("unsupported marker name: %r" % name)
        result = result or holds
    return result


def applies(marker):
    if not marker:
        return True
    results = {marker_holds(marker, python) for python in PYTHON_RANGE}
    if len(results) != 1:
        raise ValueError("marker depends on the 3.10 micro version: %r" % marker)
    return results.pop()


def parse_lock(text):
    rows = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        requirement, _, marker = line.partition(";")
        name, sep, version = requirement.strip().partition("==")
        if not sep or not name.strip() or not version.strip() or " " in name.strip():
            raise ValueError("not an exact pin: %r" % line)
        rows.append({"line": line, "name": name.strip(), "canon": canon(name.strip()), "version": version.strip(),
                     "marker": marker.strip()})
    return rows


def parse_uvlock(text):
    """{(canon name, version): [(url, sha256, size), ...]} for the registry packages of uv.lock: each [[package]]
    block's own keys, read before its first [package.*] sub-table."""
    packages = {}
    for block in text.split("\n[[package]]\n")[1:]:
        head = block.split("\n[package.", 1)[0]
        name = re.search(r'^name = "([^"]+)"$', head, re.M)
        version = re.search(r'^version = "([^"]+)"$', head, re.M)
        if not name or not version:
            continue
        key = (canon(name.group(1)), version.group(1))
        if key in packages:
            raise ValueError("uv.lock lists %s %s twice" % key)
        wheels = []
        if REGISTRY in head.splitlines():
            listing = re.search(r"^wheels = \[\n(.*?)^\]$", head, re.M | re.S)
            for row in (listing.group(1).splitlines() if listing else []):
                match = WHEEL_RE.search(row)
                if match:
                    wheels.append((match.group(1), match.group(2), int(match.group(3))))
                elif row.strip():
                    raise ValueError("unread wheel row of %s %s: %r" % (key + (row.strip()[:120],)))
        packages[key] = wheels
    return packages


def mac_platforms(version):
    """packaging.tags.mac_platforms for arm64 on macOS 11 or later, newest first."""
    platforms = []
    for major in range(version[0], 10, -1):
        platforms += ["macosx_%d_0_arm64" % major, "macosx_%d_0_universal2" % major]
    platforms += ["macosx_10_%d_universal2" % minor for minor in range(16, 3, -1)]
    return platforms


def supported_tags(version):
    """packaging.tags.sys_tags order for CPython 3.10 (abi cp310) on that macOS version, arm64."""
    platforms = mac_platforms(version)
    tags = ["cp310-cp310-" + p for p in platforms] + ["cp310-abi3-" + p for p in platforms]
    tags += ["cp310-none-" + p for p in platforms]
    for minor in range(9, 1, -1):
        tags += ["cp3%d-abi3-%s" % (minor, p) for p in platforms]
    interpreters = ["py310", "py3"] + ["py3%d" % minor for minor in range(9, -1, -1)]
    for interpreter in interpreters:
        tags += ["%s-none-%s" % (interpreter, p) for p in platforms]
    tags.append("cp310-none-any")
    tags += ["%s-none-any" % interpreter for interpreter in interpreters]
    return {tag: index for index, tag in reversed(list(enumerate(tags)))}


def wheel_tags(filename):
    parts = filename[:-len(".whl")].split("-")
    if len(parts) not in (5, 6):
        raise ValueError("not a wheel file name: %r" % filename)
    py, abi, plat = parts[-3:]
    return {"%s-%s-%s" % (a, b, c) for a in py.split(".") for b in abi.split(".") for c in plat.split(".")}


def choose(wheels, version):
    ranks = supported_tags(version)
    scored = []
    for url, sha, size in wheels:
        filename = url.rsplit("/", 1)[1]
        best = min((ranks[tag] for tag in wheel_tags(filename) if tag in ranks), default=None)
        if best is not None:
            scored.append((best, filename, url, sha, size))
    if not scored:
        return None
    scored.sort()
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        raise ValueError("two wheels share the best tag: %s, %s" % (scored[0][1], scored[1][1]))
    return scored[0][1:]


def derive(lock_bytes, uvlock_bytes):
    rows = parse_lock(lock_bytes.decode("utf-8"))
    packages = parse_uvlock(uvlock_bytes.decode("utf-8"))
    lines = [HEADER,
             "# lock requirements-runtime.lock sha256 %s" % hashlib.sha256(lock_bytes).hexdigest(),
             "# uv.lock sha256 %s" % hashlib.sha256(uvlock_bytes).hexdigest(),
             "# target CPython 3.10.x, macOS arm64 (the same choice for macOS 11.0, 16.0 and 26.0); "
             "pip's tag preference; registry https://pypi.org/simple"]
    pins, skipped, seen = [], [], set()
    for row in rows:
        if not applies(row["marker"]):
            skipped.append("# skipped (marker false here): %s" % row["line"])
            continue
        if row["canon"] in seen:
            raise ValueError("two applicable lock lines for %s" % row["name"])
        seen.add(row["canon"])
        wheels = packages.get((row["canon"], row["version"]))
        if wheels is None:
            raise ValueError("uv.lock has no registry entry for %s" % row["line"])
        choices = {choose(wheels, version) for version in MACOS_VERSIONS}
        if None in choices:
            raise ValueError("no cp310 macOS arm64 wheel in uv.lock for %s" % row["line"])
        if len(choices) != 1:
            raise ValueError("the wheel for %s depends on the macOS version: %s" % (
                row["line"], sorted(choice[0] for choice in choices)))
        filename, url, sha, size = choices.pop()
        pins.append("\t".join((row["canon"], row["version"], filename, str(size), sha, url)))
    lines += skipped
    lines.append("# applicable lines %d, wheels %d, skipped %d" % (len(pins), len(pins), len(skipped)))
    return ("\n".join(lines + pins) + "\n").encode("utf-8")


def read_pins(text):
    pins = []
    for line in text.splitlines():
        if line and not line.startswith("#"):
            name, version, filename, size, sha, url = line.split("\t")
            if "/" in filename or filename.startswith(".") or not filename.endswith(".whl"):
                raise ValueError("unsafe wheel file name: %r" % filename)
            pins.append({"name": name, "version": version, "file": filename, "size": int(size), "sha256": sha,
                         "url": url})
    return pins


def report(pins_path, lock_path, folder):
    with open(pins_path, "rb") as handle:
        pins_bytes = handle.read()
    pins = read_pins(pins_bytes.decode("utf-8"))
    with open(lock_path, "rb") as handle:
        lock_sha = hashlib.sha256(handle.read()).hexdigest()
    declared = re.search(r"^# lock requirements-runtime\.lock sha256 ([0-9a-f]{64})$", pins_bytes.decode("utf-8"), re.M)
    if not declared or declared.group(1) != lock_sha:
        raise SystemExit("the pins were derived for another lock than %s" % lock_path)
    present = sorted(os.listdir(folder))
    wanted = sorted(pin["file"] for pin in pins)
    if present != wanted:
        raise SystemExit("the folder does not hold exactly the pinned wheels: extra %s, missing %s" % (
            sorted(set(present) - set(wanted)), sorted(set(wanted) - set(present))))
    for pin in pins:
        path = os.path.join(folder, pin["file"])
        with open(path, "rb") as handle:
            data = handle.read()
        if os.path.islink(path) or len(data) != pin["size"] or hashlib.sha256(data).hexdigest() != pin["sha256"]:
            raise SystemExit("%s differs from its pin" % pin["file"])
    record = {"lock_sha256": lock_sha, "source": "PyPI wheels pinned by the candidate's uv.lock (wheelhouse.py)",
              "pins_sha256": hashlib.sha256(pins_bytes).hexdigest(), "wheels": pins}
    with open(os.path.join(folder, "wheelhouse-report.json"), "x") as handle:
        json.dump(record, handle, indent=1, sort_keys=True)
    with open(os.path.join(folder, "SHA256SUMS"), "x") as handle:
        for pin in sorted(pins, key=lambda item: item["file"]):
            handle.write("%s  %s\n" % (pin["sha256"], pin["file"]))
    print("wheels=%d lock_sha256=%s" % (len(pins), lock_sha))
    return 0


def main(argv):
    if argv[:1] == ["derive"] and len(argv) == 4:
        with open(argv[1], "rb") as lock, open(argv[2], "rb") as uvlock:
            blob = derive(lock.read(), uvlock.read())
        with open(argv[3], "xb") as handle:
            handle.write(blob)
        return 0
    if argv[:1] == ["check"] and len(argv) == 4:
        with open(argv[1], "rb") as lock, open(argv[2], "rb") as uvlock, open(argv[3], "rb") as pins:
            same = derive(lock.read(), uvlock.read()) == pins.read()
        print("pins %s" % ("reproduced" if same else "DIFFER from a fresh derivation"))
        return 0 if same else 1
    if argv[:1] == ["report"] and len(argv) == 4:
        return report(*argv[1:])
    raise SystemExit(__doc__)


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except ValueError as exc:
        sys.exit("refused: %s" % exc)

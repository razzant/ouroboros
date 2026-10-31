#!/usr/bin/env python3
"""Stage the Ubuntu 22.04 x86_64 GTK3/WebKit2 4.1 closure after PyInstaller.

Run only in the isolated Linux builder described in packaging/appimage/README.md.
No downloads or package installation here. Package versions, file hashes and
relocations are recorded in linux-gtk/manifest.json; missing inputs fail closed.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import struct
import subprocess
import sys


PACKAGES = (
    "libwebkit2gtk-4.1-0", "gir1.2-webkit2-4.1", "gir1.2-javascriptcoregtk-4.1",
    "gir1.2-gtk-3.0", "gir1.2-soup-3.0", "gir1.2-glib-2.0", "gir1.2-pango-1.0",
    "gir1.2-harfbuzz-0.0", "gir1.2-atk-1.0", "gir1.2-gdkpixbuf-2.0", "gir1.2-freedesktop",
    "libgirepository-1.0-1", "libgtk-3-0", "libgtk-3-common",
    "libgdk-pixbuf-2.0-0", "libgdk-pixbuf2.0-common", "librsvg2-common",
    "glib-networking", "gsettings-desktop-schemas", "gstreamer1.0-plugins-base",
    "gstreamer1.0-plugins-good", "libgstreamer1.0-0", "bubblewrap", "xdg-dbus-proxy",
    "adwaita-icon-theme", "hicolor-icon-theme", "shared-mime-info",
)
# libc/loader and the machine's graphics driver ABI belong to the declared OS.
# GTK, WebKit, GI, Soup, ICU, TLS and their other dependencies do not.
HOST_LIBRARIES = re.compile(
    r"^(ld-linux.*|lib(c|m|dl|pthread|rt|util|resolv|nss_.*)\.so\..*|"
    r"lib(GL|EGL|GLX|GLdispatch|gbm|drm.*)\.so\..*)$"
)
# PyInstaller and python-build-standalone each ship their own libpython.
INTERPRETER_LIBRARY = re.compile(r"^libpython3\.\d+\.so\.1\.0$")
TYPELIB_MAGIC = b"GOBJ\nMETADATA\r\n\x1a"
TRIPLET = "x86_64-linux-gnu"


def output(*args: str) -> str:
    return subprocess.check_output(args, text=True, env={
        key: value for key, value in os.environ.items() if key != "LD_LIBRARY_PATH"
    })


def elf(path: Path) -> bool:
    with path.open("rb") as stream:
        return stream.read(4) == b"\x7fELF"


def package_owner(path: str) -> str:
    candidates = [path, str(Path(path).resolve())]
    if path.startswith("/lib/"):
        candidates.append("/usr" + path)
    for candidate in candidates:
        result = subprocess.run(["dpkg-query", "-S", candidate], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.split(": ", 1)[0]
    raise RuntimeError(f"Native library has no package provenance: {path}")


def linked_libraries(source, linked: str, interpreter: Path | None) -> list[tuple[str, str]]:
    """Resolved ldd dependencies to ship; anything unresolved fails closed.

    A GI extension naming libpython is satisfied only by the copy its own
    interpreter ships; a builder libpython is never copied or aliased.
    """
    resolved = []
    for line in linked.splitlines():
        match = re.match(r"\s*(\S+) => (/\S+|not found)", line)
        if match and INTERPRETER_LIBRARY.fullmatch(match[1]):
            if interpreter is None or not (interpreter / match[1]).is_file():
                raise RuntimeError(f"{source} needs {match[1]}, which its interpreter does not ship")
        elif "not found" in line:
            raise RuntimeError(f"Unresolved dependency: {source}\n{linked}")
        elif match:
            resolved.append((match[1], match[2]))
    return resolved


def typelib_requirements(data: bytes) -> tuple[list[str], list[str]]:
    """Header dependencies and shared libraries of a format-4 GI typelib.

    gitypelib-internal.h (gobject-introspection 1.72): Header.dependencies at
    byte 36 and Header.shared_library at 52 are offsets of C strings.
    """
    if data[:16] != TYPELIB_MAGIC or data[16] != 4:
        raise RuntimeError("Unknown typelib format; the closure check must be requalified")

    def string(field: int) -> str:
        (offset,) = struct.unpack_from("<I", data, field)
        return data[offset:data.index(b"\0", offset)].decode() if offset else ""

    return ([name for name in string(36).split("|") if name],
            [name for name in string(52).split(",") if name])


def typelib_libraries(directory: Path) -> list[tuple[str, str]]:
    """(typelib, library) pairs for a typelib directory that must be closed.

    libgirepository searches GI_TYPELIB_PATH, then its compiled system typelib
    directories, and requires header dependencies eagerly. A gap here would be
    filled silently from the builder's or user's /usr, so it is refused.
    """
    named = []
    for typelib in sorted(directory.glob("*.typelib")):
        dependencies, shared = typelib_requirements(typelib.read_bytes())
        missing = [name for name in dependencies if not (directory / f"{name}.typelib").is_file()]
        if missing:
            raise RuntimeError(f"{typelib.name} requires typelibs outside the closure: {missing}")
        named += [(typelib.name, name) for name in shared]
    return named


def relocate_webkit(data: bytes) -> tuple[bytes, list[str]]:
    """Length-preserving, bounded path edits, not a global /usr substitution.

    Release WebKit ignores WEBKIT_EXEC_PATH (developer-build option). Helpers
    and resources instead resolve from the shell's private linux-gtk cwd, which
    survives exec_source_launcher. Refuse a changed upstream helper layout.
    """
    paths = [f"/usr/lib/{TRIPLET}/webkit2gtk-4.1",
             "/usr/share/webkitgtk-4.1", "/usr/libexec/webkit2gtk-4.1",
             "/usr/bin/bwrap", "/usr/bin/xdg-dbus-proxy"]
    def matches(path):
        # C string path or path prefix: never rewrite an unrelated filename
        # sharing the version prefix, nor an embedded diagnostic/URL substring.
        return re.compile(rb"(?<=\x00)" + re.escape(path.encode()) + rb"(?=[/\x00])")

    if not any(matches(path).search(data) for path in (paths[0], paths[2])):
        raise RuntimeError("WebKit helper prefix changed; relocation must be requalified")
    if not all(matches(path).search(data) for path in paths[3:]):
        raise RuntimeError("WebKit sandbox helper paths changed; relocation must be requalified")
    edits = []
    for path in paths:
        old = path.encode()
        if matches(path).search(data):
            data = matches(path).sub(b"././" + old[4:], data)
            edits.append(path)
    return data, edits


def collect_source_packages(packages) -> dict[str, str]:
    """Keep every observed source version; never silently omit matching source."""
    observed: dict[str, set[str]] = {}
    for package in packages:
        name, version = output('dpkg-query', '-W', '-f=${source:Package}\t${source:Version}', package).split('\t')
        observed.setdefault(name, set()).add(version)
    conflicts = {name: sorted(versions) for name, versions in sorted(observed.items()) if len(versions) > 1}
    if conflicts:
        raise RuntimeError('Conflicting source package versions: ' + json.dumps(conflicts, sort_keys=True))
    return {name: next(iter(versions)) for name, versions in sorted(observed.items())}


def bundle(internal: Path) -> None:
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("GTK bundle currently targets Ubuntu 22.04 x86_64 only")
    os_release = Path("/etc/os-release").read_text()
    if 'ID=ubuntu\n' not in os_release or 'VERSION_ID="22.04"' not in os_release:
        raise RuntimeError("Build the GTK closure on the declared Ubuntu 22.04 baseline")
    root = internal / "linux-gtk"
    if root.exists():
        raise RuntimeError(f"Refusing to reuse a partial GTK closure: {root}")
    root.mkdir()
    origins, libraries, pending, packages = {}, {}, [], {}

    def copy(source: Path, relative: Path) -> Path:
        target = root / relative
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target, follow_symlinks=True)
            origins[relative.as_posix()] = str(source)
            if elf(target):
                pending.append(source)
        return target

    def ship(name: str, path: Path) -> None:
        if HOST_LIBRARIES.fullmatch(name):
            return
        relative = Path("lib") / TRIPLET / name
        copy(path, relative)
        libraries[name] = relative
        owner = package_owner(str(path))
        for package in owner.split(", "):
            packages[package] = output("dpkg-query", "-W", "-f=${Version}", package)
            copyright = Path("/usr/share/doc") / package.split(":")[0] / "copyright"
            if copyright.is_file():
                copy(copyright, Path("share/doc") / package.split(":")[0] / "copyright")

    for package in PACKAGES:
        packages[package] = output("dpkg-query", "-W", "-f=${Version}", package)
        for name in output("dpkg-query", "-L", package).splitlines():
            source = Path(name)
            if not source.is_file():
                continue
            if name.startswith(("/usr/lib/", "/usr/libexec/", "/usr/share/")):
                copy(source, source.relative_to("/usr"))
            elif name in ("/usr/bin/bwrap", "/usr/bin/xdg-dbus-proxy"):
                copy(source, source.relative_to("/usr"))

    # GI dlopens typelib-named libraries by soname: ship them with the closure.
    for typelib, name in typelib_libraries(root / "lib" / TRIPLET / "girepository-1.0"):
        path = Path("/usr/lib") / TRIPLET / name
        if "/" in name or not path.exists():
            raise RuntimeError(f"{typelib} names an unshippable library: {name}")
        ship(name, path)

    # GI extensions in the frozen and source interpreters need the same closure.
    interpreters = {}
    for pattern, interpreter in (("gi/_gi*.so", internal),
                                 ("python-standalone/lib/python*/site-packages/gi/_gi*.so",
                                  internal / "python-standalone/lib")):
        extensions = list(internal.glob(pattern))
        if not extensions:
            raise RuntimeError(f"GI missing from a shipped interpreter: {pattern}")
        pending.extend(extensions)
        interpreters.update(dict.fromkeys(extensions, interpreter))
    visited = set()
    while pending:
        source = pending.pop()
        if source in visited:
            continue
        visited.add(source)
        for name, path in linked_libraries(source, output("ldd", str(source)), interpreters.get(source)):
            ship(name, Path(path))

    # Debian copyright files can refer to these rather than carry the full text.
    for license_file in Path('/usr/share/common-licenses').iterdir():
        if license_file.is_file():
            copy(license_file, Path('share/common-licenses') / license_file.name)
    source_packages = collect_source_packages(packages)

    webkit = root / "lib" / TRIPLET / "libwebkit2gtk-4.1.so.0"
    original = webkit.read_bytes()
    data, edits = relocate_webkit(original)
    # Package symlink aliases were dereferenced. Patch every alias to avoid a
    # loader choosing an unrelocated versioned filename.
    for target in webkit.parent.glob("libwebkit2gtk-4.1.so.*"):
        target.write_bytes(data)
    libraries[webkit.name] = webkit.relative_to(root)
    for name, relative in libraries.items():
        target = internal / name
        target.unlink(missing_ok=True)
        target.symlink_to(Path("linux-gtk") / relative)
    # Apply the same host ABI boundary to anything Analysis collected first.
    for target in internal.glob("*.so*"):
        if HOST_LIBRARIES.fullmatch(target.name):
            target.unlink()

    schemas = root / "share/glib-2.0/schemas"
    subprocess.run(["glib-compile-schemas", str(schemas)], check=True)
    subprocess.run(["gio-querymodules", str(root / "lib" / TRIPLET / "gio/modules")], check=True)
    subprocess.run(["update-mime-database", str(root / "share/mime")], check=True)
    immodules = root / "lib" / TRIPLET / "gtk-3.0/3.0.0"
    imquery = f"/usr/lib/{TRIPLET}/libgtk-3-0/gtk-query-immodules-3.0"
    imcache = output(imquery, *map(str, sorted((immodules / "immodules").glob("*.so"))))
    (immodules / "immodules.cache").write_text(imcache.replace(str(root), "."))
    loaders = root / "lib" / TRIPLET / "gdk-pixbuf-2.0/2.10.0"
    query = f"/usr/lib/{TRIPLET}/gdk-pixbuf-2.0/gdk-pixbuf-query-loaders"
    cache = output(query, *map(str, sorted((loaders / "loaders").glob("*.so"))))
    (loaders / "loaders.cache").write_text(cache.replace(str(root), "."))
    for required in ("Gtk-3.0.typelib", "WebKit2-4.1.typelib", "Soup-3.0.typelib"):
        if not (root / "lib" / TRIPLET / "girepository-1.0" / required).is_file():
            raise RuntimeError(f"Missing typelib: {required}")
    for helper in ("WebKitWebProcess", "WebKitNetworkProcess"):
        if not os.access(root / "lib" / TRIPLET / "webkit2gtk-4.1" / helper, os.X_OK):
            raise RuntimeError(f"Missing executable WebKit helper: {helper}")
    shutil.copy2("packaging/appimage/desktop-env.sh", root / "desktop-env.sh")
    shutil.copy2(__file__, root / 'share/doc/webkit-relocation.py')
    (root / 'share/doc/webkit-relocation.txt').write_text(
        'Ouroboros modifies the WebKit shared library by equal-length path relocation.\n'
        'Modification build time (UTC): ' + datetime.now(timezone.utc).isoformat() + '\n'
        'The exact transformation is webkit-relocation.py:relocate_webkit.\n'
        'Original SHA256: ' + hashlib.sha256(original).hexdigest() + '\n'
        'Modified SHA256: ' + hashlib.sha256(data).hexdigest() + '\n'
        'Corresponding Debian source package: ' + json.dumps({k: v for k, v in source_packages.items() if 'webkit' in k}) + '\n'
        'Package notices and common license texts accompany this payload.\n'
        'Source archives/build inputs must be retained and made available before distribution;\n'
        'this manifest alone is not corresponding source or a compliance determination.\n')
    manifest = {
        "baseline": "ubuntu-22.04-x86_64", "packages": packages, "source_packages": source_packages,
        "webkit_original_sha256": hashlib.sha256(original).hexdigest(),
        "relocated_webkit_prefixes": edits, "host_library_pattern": HOST_LIBRARIES.pattern,
        "files": {str(p.relative_to(root)): {
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "source": origins.get(str(p.relative_to(root)), "generated"),
        } for p in sorted(root.rglob("*")) if p.is_file()},
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    install_desktop_entry(internal.parent)


def install_desktop_entry(payload: Path) -> None:
    """All payload consumers save caller state before PyInstaller changes it."""
    binary = payload / "Ouroboros.bin"
    if binary.exists():
        raise RuntimeError(f"Refusing to overwrite a desktop entry: {binary}")
    (payload / "Ouroboros").rename(binary)
    shutil.copy2("packaging/appimage/desktop-launcher.sh", payload / "Ouroboros")
    (payload / "Ouroboros").chmod(0o755)


if __name__ == "__main__":
    bundle(Path(sys.argv[1]).resolve())

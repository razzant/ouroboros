#!/bin/bash
# Wrap the PyInstaller Linux payload (dist/Ouroboros) into native .deb and .rpm
# packages, so Debian/Ubuntu and Fedora/RHEL users can install with their own
# package manager instead of unpacking the tarball by hand.
#
# Both packages carry the exact same payload as the tarball, installed to
# /opt/ouroboros with a /usr/bin/ouroboros symlink, a desktop entry, and an
# opt-in systemd user unit.
#
# Usage: bash build_linux_packages.sh [payload-dir] [output-dir]
set -euo pipefail

VERSION=$(tr -d '[:space:]' < VERSION)
# Keep every accepted author-facing prerelease spelling upgrade-safe in both
# package managers. Filenames retain VERSION so they stay derivable from the
# release tag; package metadata uses the shared release grammar's canonical
# ``~rcN`` / ``~aN`` / ``~bN`` spelling.
PKG_VERSION="$(python3 -c 'import runpy, sys; release_sync = runpy.run_path("ouroboros/tools/release_sync.py"); print(release_sync["normalize_linux_package_version"](sys.argv[1]))' "$VERSION")"
PAYLOAD="${1:-dist/Ouroboros}"
OUTDIR="${2:-dist}"
DEB_PATH="$OUTDIR/ouroboros_${VERSION}_amd64.deb"
RPM_PATH="$OUTDIR/ouroboros-${VERSION}-1.x86_64.rpm"
# RED OS 8 identifies itself as platform:red80 and its rpm stamps that dist tag.
# The payload is the same self-contained bundle; the separate artifact carries
# the distro's own release tag so dnf orders and upgrades it correctly, and it
# is the one proven by an install smoke on RED OS itself.
RPM_RED80_PATH="$OUTDIR/ouroboros-${VERSION}-1.red80.x86_64.rpm"
MAINTAINER="Ouroboros maintainers <razzant@users.noreply.github.com>"
SUMMARY="Self-creating agent with constitution, background consciousness, and persistent identity"
HOMEPAGE="https://github.com/razzant/ouroboros"

echo "=== Building Linux packages for Ouroboros v${VERSION} ==="

if [ ! -d "$PAYLOAD" ]; then
    echo "ERROR: payload directory not found: $PAYLOAD" >&2
    echo "Run first: bash build_linux.sh" >&2
    exit 1
fi
for tool in dpkg-deb rpmbuild; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "ERROR: $tool is required (Debian/Ubuntu: apt-get install dpkg-dev rpm)" >&2
        exit 1
    fi
done

# Staged inside the output directory so `cp -al` can hardlink the ~GB payload
# instead of copying it three times over (build host disk is the constraint).
# Absolute: rpmbuild resolves a relative --define _topdir against /, which puts
# its buildroot on another filesystem and breaks the hardlinks.
mkdir -p "$OUTDIR"
STAGE="$(cd "$OUTDIR" && pwd)/.package-stage"
ROOT="$STAGE/root"
rm -rf "$STAGE"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$ROOT/opt" "$ROOT/usr/bin" "$ROOT/usr/share/applications" \
    "$ROOT/usr/share/pixmaps" "$ROOT/usr/lib/systemd/user"

if ! cp -al "$PAYLOAD" "$ROOT/opt/ouroboros" 2>/dev/null; then
    echo "Hardlink staging is unavailable; copying the payload once instead."
    rm -rf "$ROOT/opt/ouroboros"
    cp -a "$PAYLOAD" "$ROOT/opt/ouroboros"
fi
# The packaged CLI shim resolves its own symlink before locating the bundle
# root, so a plain /usr/bin symlink is all either package needs.
ln -s /opt/ouroboros/bin/ouroboros "$ROOT/usr/bin/ouroboros"
cp assets/icon_1024.png "$ROOT/usr/share/pixmaps/ouroboros.png"
# systemd USER unit, not a system one: state lives in $HOME/Ouroboros and the
# desktop build needs the user's session. Shipping it only provides a stable
# unit name for explicit user control. The packages contain no maintainer
# script that enables or starts it during installation.
install -m 644 packaging/systemd/ouroboros.service.in \
    "$ROOT/usr/lib/systemd/user/ouroboros.service"
cat > "$ROOT/usr/share/applications/ouroboros.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Ouroboros
Comment=$SUMMARY
Exec=/opt/ouroboros/Ouroboros
Icon=ouroboros
Terminal=false
Categories=Development;Utility;
EOF

echo "--- Building $DEB_PATH ---"
INSTALLED_KB=$(du -sk "$ROOT" | cut -f1)
mkdir -p "$ROOT/DEBIAN"
cat > "$ROOT/DEBIAN/control" <<EOF
Package: ouroboros
Version: $PKG_VERSION
Architecture: amd64
Maintainer: $MAINTAINER
Section: utils
Priority: optional
Depends: git
Installed-Size: $INSTALLED_KB
Homepage: $HOMEPAGE
Description: $SUMMARY
 Ships a self-contained Python, Node.js and browser runtime under
 /opt/ouroboros; no system Python is required.
EOF
# gzip rather than the default xz: the payload is already-compressed binaries,
# so xz costs minutes of CI time for a few percent.
dpkg-deb --root-owner-group -Zgzip --build "$ROOT" "$DEB_PATH"
rm -rf "$ROOT/DEBIAN"

echo "--- Building $RPM_PATH ---"
TOPDIR="$STAGE/rpmbuild"
mkdir -p "$TOPDIR"
cat > "$STAGE/ouroboros.spec" <<EOF
# The payload is a prebuilt, self-contained bundle: no stripping, no
# debuginfo, no build-id links, and no dependency scanning of the thousands of
# vendored ELF files (which would otherwise take minutes and pin the package
# to the build host's library set).
%global __os_install_post %{nil}
%global debug_package %{nil}
%global _build_id_links none
%define _binary_payload w1.gzdio

Name:           ouroboros
Version:        $PKG_VERSION
# rel_suffix is left undefined for the generic build and set per
# distro-targeted rebuild. The build host's own dist tag is deliberately not
# used: it would stamp the builder's distro onto the generic package.
Release:        1%{?rel_suffix}
Summary:        $SUMMARY
License:        MIT
URL:            $HOMEPAGE
BuildArch:      x86_64
AutoReqProv:    no
Requires:       git

%description
$SUMMARY
Ships a self-contained Python, Node.js and browser runtime under
/opt/ouroboros; no system Python is required.

%install
rm -rf %{buildroot}
mkdir -p \
  "%{buildroot}/opt" \
  "%{buildroot}/usr/bin" \
  "%{buildroot}/usr/share/applications" \
  "%{buildroot}/usr/share/pixmaps" \
  "%{buildroot}/usr/lib/systemd/user"
# Keep the multi-gigabyte payload hardlinked inside the output-local stage,
# but recreate the absolute CLI symlink explicitly. Some rpmbuild/cp
# combinations try to hardlink its missing target when copying the whole root.
cp -al "$ROOT/opt/ouroboros" "%{buildroot}/opt/ouroboros"
ln -s /opt/ouroboros/bin/ouroboros "%{buildroot}/usr/bin/ouroboros"
cp -a "$ROOT/usr/share/applications/ouroboros.desktop" \
  "%{buildroot}/usr/share/applications/ouroboros.desktop"
cp -a "$ROOT/usr/share/pixmaps/ouroboros.png" \
  "%{buildroot}/usr/share/pixmaps/ouroboros.png"
cp -a "$ROOT/usr/lib/systemd/user/ouroboros.service" \
  "%{buildroot}/usr/lib/systemd/user/ouroboros.service"

%files
%defattr(-,root,root,-)
/opt/ouroboros
/usr/bin/ouroboros
/usr/share/applications/ouroboros.desktop
/usr/share/pixmaps/ouroboros.png
/usr/lib/systemd/user/ouroboros.service

%changelog
EOF
# Both rpms come off the same staged payload, so the second rpmbuild only costs
# the compression pass.
build_rpm() {
    local suffix="$1" destination="$2"
    local defines=(--define "_topdir $TOPDIR")
    # rpmbuild rejects a --define with an empty body, so the generic build
    # simply leaves %{?rel_suffix} undefined.
    if [ -n "$suffix" ]; then
        defines+=(--define "rel_suffix $suffix")
    fi
    rpmbuild -bb --quiet "${defines[@]}" "$STAGE/ouroboros.spec"
    mv "$TOPDIR/RPMS/x86_64/ouroboros-${PKG_VERSION}-1${suffix}.x86_64.rpm" "$destination"
}
build_rpm "" "$RPM_PATH"

echo "--- Building $RPM_RED80_PATH ---"
build_rpm ".red80" "$RPM_RED80_PATH"

echo ""
echo "=== Done ==="
ls -lh "$DEB_PATH" "$RPM_PATH" "$RPM_RED80_PATH"
echo ""
echo "Install:  sudo apt install ./$(basename "$DEB_PATH")"
echo "          sudo dnf install ./$(basename "$RPM_PATH")"
echo "          sudo dnf install ./$(basename "$RPM_RED80_PATH")   # RED OS 8"

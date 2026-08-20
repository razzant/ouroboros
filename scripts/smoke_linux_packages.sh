#!/bin/bash
# Install the built .deb and .rpm in stock distro containers and check that the
# installed systemd unit, packaged CLI, and desktop launcher after a real install.
#
# Two lanes, because they carry different risk:
#
#   official  Docker Hub images. This lane gates the release, so every package
#             — including the RED OS one, which is an ordinary rpm — has to
#             prove it installs and runs here.
#   vendor    Astra Linux and RED OS images from the vendors' own registries.
#             This lane confirms the packages on the OS they target, but its
#             registries are third-party and their reachability from CI is
#             outside our control, so it runs informationally and must never
#             hold back a tagged release.
#
# Docker Hub official images are tag-pinned on purpose: they are the smoke
# environment, not a build input, and the point is to track what users actually
# run. The vendor images are pinned by digest instead.
#
# The single .deb covers Astra Linux too: Astra is Debian-based
# (ID_LIKE=debian, dpkg 1.21) and installs the same package, so it needs a
# smoke target rather than a build of its own.
#
# Usage: bash smoke_linux_packages.sh <official|vendor> <deb> <rpm> <red80-rpm>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LANE="$1"
DEB="$2"
RPM="$3"
RPM_RED80="$4"
ASTRA_IMAGE="registry.astralinux.ru/library/astra/ubi18@sha256:694fcfd48cf152ec833caeb63dba416e7ea55d8491bf5b46dd6c29d6fbf0ede3"
RED80_IMAGE="registry.red-soft.ru/ubi8/ubi@sha256:cae37cb16daadfecae09e854471592f27bcd6aefb4b44da1e5b22bba57b1e9cd"

smoke_package() {
    local image="$1" package="$2" install_cmd="$3"
    local name
    name="$(basename "$package")"
    echo "--- Smoking $name in $image ---"
    docker run --rm \
        --volume "$(cd "$(dirname "$package")" && pwd)/$name:/tmp/$name:ro" \
        --volume "$SCRIPT_DIR/betterleaks_platform_smoke.py:/tmp/betterleaks_platform_smoke.py:ro" \
        "$image" sh -c "
            set -eu
            $install_cmd /tmp/$name
            command -v git >/dev/null
            test -f /usr/share/applications/ouroboros.desktop
            test -f /usr/share/pixmaps/ouroboros.png
            test -s /usr/lib/systemd/user/ouroboros.service
            grep -Fqx 'ExecStart=/opt/ouroboros/Ouroboros' \
              /usr/lib/systemd/user/ouroboros.service
            grep -Fqx 'KillMode=control-group' \
              /usr/lib/systemd/user/ouroboros.service
            ! grep -q '^Restart=' /usr/lib/systemd/user/ouroboros.service
            test -x /opt/ouroboros/Ouroboros
            ouroboros --help >/dev/null
            PYTHONPATH=/opt/ouroboros/_internal \
              /opt/ouroboros/_internal/python-standalone/bin/python3 \
              /tmp/betterleaks_platform_smoke.py \
              --bundle-root /opt/ouroboros/_internal
            mkdir -p /tmp/ouroboros-smoke-home /tmp/ouroboros-smoke-data
            set +e
            HOME=/tmp/ouroboros-smoke-home \
              XDG_CACHE_HOME=/tmp/ouroboros-smoke-home/.cache \
              OUROBOROS_DATA_DIR=/tmp/ouroboros-smoke-data \
              timeout --signal=TERM --kill-after=5s 5s /opt/ouroboros/Ouroboros
            launcher_rc=\$?
            set -e
            if [ \"\$launcher_rc\" -ne 124 ]; then
                echo \"desktop launcher exited before the smoke deadline (rc=\$launcher_rc)\" >&2
                exit 1
            fi
            test -s /tmp/ouroboros-smoke-data/logs/launcher.log
        "
}

case "$LANE" in
    official)
        smoke_package ubuntu:22.04 "$DEB" "apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq"
        smoke_package fedora:42 "$RPM" "dnf install -y -q"
        smoke_package fedora:42 "$RPM_RED80" "dnf install -y -q"
        ;;
    vendor)
        smoke_package "$ASTRA_IMAGE" "$DEB" "apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq"
        smoke_package "$RED80_IMAGE" "$RPM_RED80" "dnf install -y -q"
        ;;
    *)
        echo "ERROR: unknown lane: $LANE (expected 'official' or 'vendor')" >&2
        exit 2
        ;;
esac

echo "=== Linux package smoke passed ($LANE lane) ==="

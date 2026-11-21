#!/bin/bash
# Install the built .deb and .rpm in stock distro containers and check that the
# installed systemd unit, packaged CLI, and desktop launcher after a real install.
#
# Two kinds of lane, because they carry different risk:
#
#   official  Docker Hub images. This lane gates the release, so every package
#             — including the RED OS one, which is an ordinary rpm — has to
#             prove it installs and runs here.
#   astra,    Astra Linux and RED OS images from the vendors' own registries.
#   redos     These lanes confirm the packages on the OS they target, but their
#             registries are third-party and their reachability from CI is
#             outside our control, so each runs informationally, as its own CI
#             leg, and must never hold back a tagged release.
#
# Docker Hub official images are tag-pinned on purpose: they are the smoke
# environment, not a build input, and the point is to track what users actually
# run. The vendor lanes use pinned linux/amd64 images. On a cache miss only
# redos-ci uses the hosted runner's single-registry downloader.
#
# The single .deb covers Astra Linux too: Astra is Debian-based
# (ID_LIKE=debian, dpkg 1.21) and installs the same package, so it needs a
# smoke target rather than a build of its own.
#
# Usage: bash smoke_linux_packages.sh official <deb> <rpm> <red80-rpm>
#        bash smoke_linux_packages.sh astra <deb>
#        bash smoke_linux_packages.sh redos <red80-rpm>
#        bash smoke_linux_packages.sh vendor <deb> <rpm> <red80-rpm>  # both vendor checks, in order
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
LANE="$1"
ASTRA_IMAGE="registry.astralinux.ru/library/astra/ubi18@sha256:694fcfd48cf152ec833caeb63dba416e7ea55d8491bf5b46dd6c29d6fbf0ede3"

smoke_package() {
    local image="$1" package="$2" install_cmd="$3"
    local name="${package##*/}" directory pull=missing
    directory="$(CDPATH= cd -P -- "$(dirname -- "$package")" && pwd -P)"
    local file="$directory/$name"
    if [[ ! -f "$file" || ! -r "$file" ]]; then
        echo "ERROR: expected a readable package file: $package" >&2
        return 1
    fi
    if [[ "$LANE" != official ]]; then
        image="$(vendor_image_id "$1" "${4:-}")"
        pull=never
    fi
    echo "--- Smoking $name in $image ---"
    # --mount fails if the file disappears. Quote its CSV fields for literal commas and quotes.
    local mount_source="${file//\"/\"\"}" mount_name="${name//\"/\"\"}"
    docker run --rm --pull="$pull" \
        --mount "type=bind,\"source=$mount_source\",\"target=/tmp/$mount_name\",readonly" \
        --volume "$SCRIPT_DIR/betterleaks_platform_smoke.py:/tmp/betterleaks_platform_smoke.py:ro" \
        "$image" sh -c "
            set -eu
            $install_cmd \"\$1\" || exit \$?
            command -v git >/dev/null
            test -f /usr/share/applications/ouroboros.desktop
            test -f /usr/share/pixmaps/ouroboros.png
            test -s /usr/lib/systemd/user/ouroboros.service
            grep -Fqx 'ExecStart=/opt/ouroboros/Ouroboros --launch-intent automatic' \
              /usr/lib/systemd/user/ouroboros.service
            grep -Fqx 'KillMode=control-group' \
              /usr/lib/systemd/user/ouroboros.service
            restart_rc=0
            grep -q '^Restart=' /usr/lib/systemd/user/ouroboros.service || restart_rc=\$?
            test \"\$restart_rc\" -eq 1  # no match; a read error must fail too
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
        " sh "/tmp/$name"
}

vendor_image_id() {
    local reference="$1" config="$2" source="$1" metadata platform id digests
    if ! docker image inspect "$reference" >/dev/null 2>&1; then
        if [[ -n "$config" ]] && docker image inspect "$config" >/dev/null 2>&1; then
            source="$config"
        elif [[ "$LANE" == redos-ci ]]; then
            python3 -I "$SCRIPT_DIR/ci_redos_image.py" | docker image load >&2 || return
            source="localhost/redos-smoke:${config#sha256:}"
        else
            docker pull "$reference" >&2 || return
        fi
    fi
    if [[ "$source" == "$reference" ]]; then
        digests="$(docker image inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' "$source")" || return
        if ! grep -Fxq -e "$reference" <<< "$digests"; then
            echo "ERROR: expected $reference in local RepoDigests" >&2
            return 1
        fi
    fi
    # A loaded archive can lack RepoDigests. Resolve the verified import to
    # Docker's immutable local ID, which can differ between image stores.
    metadata="$(docker image inspect --format '{{.Os}}/{{.Architecture}} {{.Id}}' "$source")" || return
    read -r platform id <<< "$metadata"
    if [[ "$platform" != linux/amd64 || ! "$id" =~ ^sha256:[0-9a-f]{64}$ || ( "$source" == "$config" && "$id" != "$config" ) ]]; then
        echo "ERROR: unexpected local image for $reference: $metadata" >&2
        return 1
    fi
    echo "--- Verified $platform $reference ($id) ---" >&2
    printf '%s\n' "$id"
}

case "$LANE" in
    official)
        DEB="$2" RPM="$3" RPM_RED80="$4"
        smoke_package ubuntu:22.04 "$DEB" "apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq"
        smoke_package fedora:42 "$RPM" "dnf install -y -q"
        smoke_package fedora:42 "$RPM_RED80" "dnf install -y -q"
        ;;
    vendor)
        "$BASH" "$0" astra "$2"
        "$BASH" "$0" redos "$4"
        ;;
    astra)
        smoke_package "$ASTRA_IMAGE" "$2" "apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq"
        ;;
    redos|redos-ci)
        # RED OS 8 build 8.0.3-261103, manifest reference and matching config ID.
        read -r red80_image red80_id < "$SCRIPT_DIR/redos-ubi8.pin"
        smoke_package "$red80_image" "$2" "dnf install -y -q" "$red80_id"
        ;;
    *)
        echo "ERROR: unknown lane: $LANE (expected 'official', 'vendor', 'astra' or 'redos')" >&2
        exit 2
        ;;
esac

echo "=== Linux package smoke passed ($LANE lane) ==="

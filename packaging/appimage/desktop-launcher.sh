#!/bin/sh
# Public entry shared by AppImage, tarball and /opt/ouroboros (deb/rpm).
set -eu
ENTRY=$0
while [ -L "$ENTRY" ]; do
    DIRECTORY=$(CDPATH= cd -- "$(dirname -- "$ENTRY")" && pwd)
    LINK=$(readlink "$ENTRY")
    case "$LINK" in /*) ENTRY=$LINK ;; *) ENTRY=$DIRECTORY/$LINK ;; esac
done
PAYLOAD=$(CDPATH= cd -- "$(dirname -- "$ENTRY")" && pwd)
export OUROBOROS_BUNDLE_DIR="$PAYLOAD/_internal"
. "$OUROBOROS_BUNDLE_DIR/linux-gtk/desktop-env.sh"
exec "$PAYLOAD/Ouroboros.bin" "$@"

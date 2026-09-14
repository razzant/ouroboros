#!/bin/sh
# Ubuntu's signed snapshot is configured by the USB installer. No Android init replacement.
set -eu
ensure_localhost() {
    python3 - "$1" <<'PY'
from pathlib import Path
import ipaddress
import sys

path = Path(sys.argv[1])
existed = path.exists()
original = path.read_bytes() if existed else b""
present = set()
for line in original.splitlines():
    fields = line.split(b"#", 1)[0].split()
    if len(fields) > 1 and b"localhost" in [name.lower() for name in fields[1:]]:
        try:
            present.add(str(ipaddress.ip_address(fields[0].decode("ascii"))))
        except ValueError:
            pass
missing = [line for address, line in (
    ("127.0.0.1", b"127.0.0.1 localhost\n"),
    ("::1", b"::1 localhost ip6-localhost ip6-loopback\n"),
) if address not in present]
if missing:
    with path.open("ab") as stream:
        if original and not original.endswith(b"\n"):
            stream.write(b"\n")
        stream.writelines(missing)
    if not existed:
        path.chmod(0o644)
print("Localhost mappings added:", len(missing))
PY
}
if [ "${1:-}" = "--ensure-localhost" ]; then
    ensure_localhost "${2:-/etc/hosts}"
    exit 0
fi
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip python3-dev git curl ca-certificates \
    build-essential cmake ninja-build bison flex pkg-config libfmt-dev libgtest-dev \
    libexpat1-dev libpng-dev libprotobuf-dev protobuf-compiler zlib1g-dev \
    openjdk-17-jdk-headless util-linux procps patch unzip
ensure_localhost /etc/hosts

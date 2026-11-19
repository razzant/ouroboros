"""Write the pinned RED OS 8 Docker load archive to stdout on hosted GitHub CI."""

import gzip
import hashlib
import http.client
import json
import os
import re
import ssl
import sys
import tarfile
import tempfile
from contextlib import closing
from pathlib import Path


HOST = "registry.red-soft.ru"
REPO = "ubi8/ubi"
MANIFEST_TYPE = "application/vnd.docker.distribution.manifest.v2+json"
HERE = Path(__file__).resolve().parent
CA = HERE.parent / ".github/certs/registry.red-soft.ru-2024.crt"
CA_SHA256 = "6f9d829c8e6712444fce3624658d8788672849c5d5b7b53fd9cf7e83eac4193e"


def registry_connection():
    if os.environ.get("GITHUB_ACTIONS") != "true" or os.environ.get("RUNNER_ENVIRONMENT") != "github-hosted":
        raise ValueError("The RED OS registry CA requires a GitHub-hosted runner")
    pem = CA.read_bytes()
    if hashlib.sha256(pem).hexdigest() != CA_SHA256:
        raise ValueError("RED OS issuing CA digest mismatch")
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.verify_flags |= ssl.VERIFY_X509_PARTIAL_CHAIN
    context.load_verify_locations(cadata=pem.decode("ascii"))
    return http.client.HTTPSConnection(HOST, 443, context=context, timeout=60)


def get(connection, path, headers):
    # HTTPSConnection sends one request to its own peer. Only a complete 200 is
    # usable here. Neither Location nor an authentication challenge supplies a URL.
    connection.request("GET", path, headers=headers)
    response = connection.getresponse()
    if response.status != 200:
        response.close()
        raise ValueError(f"RED OS registry returned HTTP {response.status} for {path}")
    return response


def copy_verified(source, output, digest, size=None):
    checksum = hashlib.sha256()
    with output.open("wb") as dest:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            if size is not None and dest.tell() + len(block) > size:
                raise ValueError("RED OS blob exceeds its pinned size")
            dest.write(block)
            checksum.update(block)
        if (size is not None and dest.tell() != size) or "sha256:" + checksum.hexdigest() != digest:
            raise ValueError(f"RED OS content mismatch for {digest}")


def write_image(connection, directory, manifest_digest, config_id, output):
    with get(connection, "/service/token?service=harbor-registry&scope=repository:ubi8/ubi:pull", {}) as response:
        token = json.loads(response.read(64 * 1024))["token"]
    if not isinstance(token, str) or not re.fullmatch(r"[A-Za-z0-9._~+/-]+=*", token):
        raise ValueError("Invalid RED OS anonymous pull token")
    auth = {"Authorization": "Bearer " + token}
    with get(connection, f"/v2/{REPO}/manifests/{manifest_digest}", {**auth, "Accept": MANIFEST_TYPE}) as response:
        raw = response.read(1024 * 1024)
    if "sha256:" + hashlib.sha256(raw).hexdigest() != manifest_digest:
        raise ValueError("RED OS manifest digest mismatch")
    manifest = json.loads(raw)
    if (manifest["schemaVersion"], manifest["mediaType"], manifest["config"]["digest"]) != (2, MANIFEST_TYPE, config_id):
        raise ValueError("Unexpected RED OS manifest or config")

    def blob(descriptor, dest):
        digest = descriptor["digest"]
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
            raise ValueError("Invalid RED OS blob digest")
        with get(connection, f"/v2/{REPO}/blobs/{digest}", auth) as response:
            copy_verified(response, dest, digest, descriptor["size"])

    config = directory / (config_id.removeprefix("sha256:") + ".json")
    blob(manifest["config"], config)
    metadata = json.loads(config.read_bytes())
    if (metadata["os"], metadata["architecture"]) != ("linux", "amd64"):
        raise ValueError("Expected a linux/amd64 RED OS image")

    layers = []
    compressed = directory / "layer.gz"
    for i, (descriptor, diff_id) in enumerate(zip(manifest["layers"], metadata["rootfs"]["diff_ids"], strict=True)):
        blob(descriptor, compressed)
        layer = directory / f"layer-{i}.tar"
        with gzip.open(compressed, "rb") as stream:
            copy_verified(stream, layer, diff_id)
        layers.append(layer.name)

    # docker load restores the original config and ordered, uncompressed layer
    # tars. A content-specific local tag accommodates either Docker image store.
    index = directory / "manifest.json"
    item = {"Config": config.name, "Layers": layers, "RepoTags": [f"localhost/redos-smoke:{config.stem}"]}
    index.write_text(json.dumps([item]), encoding="utf-8")
    with tarfile.open(fileobj=output, mode="w|") as archive:
        for name in [index.name, config.name, *layers]:
            archive.add(directory / name, arcname=name)


def main():
    reference, config_id = (HERE / "redos-ubi8.pin").read_text(encoding="ascii").split()
    repository, manifest_digest = reference.split("@")
    if repository != f"{HOST}/{REPO}" or not all(
        re.fullmatch(r"sha256:[0-9a-f]{64}", value) for value in (manifest_digest, config_id)
    ):
        raise ValueError("Invalid RED OS image pin")
    with closing(registry_connection()) as connection, tempfile.TemporaryDirectory(dir=os.environ["RUNNER_TEMP"]) as tmp:
        write_image(connection, Path(tmp), manifest_digest, config_id, sys.stdout.buffer)


if __name__ == "__main__":
    main()

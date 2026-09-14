"""Run the real Java DNS serializer and atomic writer without Android or root."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest


pytestmark = pytest.mark.serial


@pytest.fixture
def dns_java(tmp_path):
    java_home = os.environ.get("JAVA_HOME")
    javac = str(Path(java_home) / "bin/javac") if java_home else shutil.which("javac")
    java = str(Path(java_home) / "bin/java") if java_home else shutil.which("java")
    if not javac or not java:
        pytest.skip("The native compiler check requires the build's JDK")
    # RuntimeClient's HTTP methods are not exercised here; only its DNS and
    # shell writers run. The complete host also compiles against the real SDK.
    stub = tmp_path / "org/json/JSONObject.java"
    stub.parent.mkdir(parents=True)
    stub.write_text('package org.json; public class JSONObject { public JSONObject() {} '
                    'public JSONObject(String s) {} public String toString() { return "{}"; } }')
    package = tmp_path / "ai/ouroboros/android"
    package.mkdir(parents=True)
    client = Path(__file__).resolve().parents[1] / "host/src/ai/ouroboros/android/RuntimeClient.java"
    shutil.copy2(client, package / client.name)
    harness = package / "DnsTest.java"
    harness.write_text('''package ai.ouroboros.android;
import java.net.InetAddress;
import java.util.Arrays;
import java.util.Collections;
public class DnsTest {
    public static void main(String[] args) throws Exception {
        if (args[0].equals("command")) {
            System.out.print(RuntimeClient.dnsWriteCommand(args[1]));
        } else if (args[0].equals("empty")) {
            System.out.print(RuntimeClient.dnsConfiguration(Collections.emptyList()));
        } else {
            InetAddress v4 = InetAddress.getByAddress(new byte[]{1, 1, 1, 1});
            InetAddress v6 = InetAddress.getByAddress(new byte[]{0x20, 1, 0x0d, (byte)0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x53});
            System.out.print(RuntimeClient.dnsConfiguration(Arrays.asList(v4, v6, v4)));
        }
    }
}
''')
    subprocess.run([javac, "-encoding", "UTF-8", "-d", str(tmp_path), str(stub), str(package / client.name), str(harness)],
                   check=True, capture_output=True)

    def invoke(*args):
        return subprocess.check_output([java, "-cp", str(tmp_path), "ai.ouroboros.android.DnsTest", *args])

    return invoke


def test_numeric_dns_preserves_order_and_deduplicates_without_resolving_names(dns_java):
    assert dns_java("servers") == b"nameserver 1.1.1.1\nnameserver 2001:db8:0:0:0:0:0:53\n"
    assert dns_java("empty") == b""


def test_resolver_write_is_atomic_idempotent_and_keeps_last_file_on_empty(dns_java, tmp_path):
    parent = tmp_path / "owner's files"
    parent.mkdir()
    target = parent / "resolv.conf"
    target.write_bytes(b"nameserver 192.0.2.1\n")
    command = dns_java("command", str(target)).decode()

    def write(value):
        subprocess.run(["sh", "-c", command], input=value, check=True, capture_output=True)

    previous = target.stat()
    write(target.read_bytes())
    assert target.stat().st_ino == previous.st_ino
    assert target.stat().st_mtime_ns == previous.st_mtime_ns
    with target.open("rb") as old_reader:
        write(b"nameserver 198.51.100.2\n")
        assert old_reader.read() == b"nameserver 192.0.2.1\n"
        assert target.read_bytes() == b"nameserver 198.51.100.2\n"
    previous = target.stat()
    write(b"")
    assert target.read_bytes() == b"nameserver 198.51.100.2\n"
    assert target.stat().st_ino == previous.st_ino
    assert target.stat().st_mode & 0o777 == 0o644
    assert not list(parent.glob("resolv.conf.ouroboros.*"))

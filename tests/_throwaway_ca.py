"""A throwaway certificate authority and the loopback leaf it signs, for tests of TLS trust.

``throwaway_ca(directory, name)`` writes the CA (``throwaway-ca.pem``) and a leaf for ``127.0.0.1`` only
(``leaf.pem``, ``leaf-key.pem``); ``file_in_capath(ca_pem, directory)`` files a CA under OpenSSL's subject-hash
name, as an ``SSL_CERT_DIR`` holds it; ``trusted_certificate(ca_pem, reject=...)`` restates a CA as OpenSSL's
``TRUSTED CERTIFICATE`` that trusts it for TLS servers, or rejects it for them. Nothing here touches the system's trust.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import ipaddress
import pathlib

import pytest


def throwaway_ca(directory: pathlib.Path, name: str = "ouroboros throwaway test ca"):
    """``(ca_pem, cert_pem, key_pem)`` under ``directory``; ``name`` (lowercase ASCII, single spaces) is the CA's
    common name, so its canonical form for ``file_in_capath`` is its own encoding."""
    pytest.importorskip("cryptography", reason="cryptography is not installed")
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    now = _dt.datetime.now(_dt.timezone.utc)
    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name)])
    # Python 3.13 enables strict X.509 verification: CA SKI/key usage and leaf AKI are required.
    ca = (
        x509.CertificateBuilder().subject_name(ca_name).issuer_name(ca_name)
        .public_key(ca_key.public_key()).serial_number(x509.random_serial_number())
        .not_valid_before(now - _dt.timedelta(days=1)).not_valid_after(now + _dt.timedelta(days=2))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False)
        .add_extension(x509.KeyUsage(
            digital_signature=False, content_commitment=False, key_encipherment=False,
            data_encipherment=False, key_agreement=False, key_cert_sign=True, crl_sign=True,
            encipher_only=False, decipher_only=False,
        ), critical=True)
        .sign(ca_key, hashes.SHA256())
    )
    leaf_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    leaf = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "127.0.0.1")]))
        .issuer_name(ca_name).public_key(leaf_key.public_key()).serial_number(x509.random_serial_number())
        .not_valid_before(now - _dt.timedelta(days=1)).not_valid_after(now + _dt.timedelta(days=2))
        .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]), critical=False)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
        .sign(ca_key, hashes.SHA256())
    )
    directory.mkdir(parents=True, exist_ok=True)
    ca_pem = directory / "throwaway-ca.pem"
    ca_pem.write_bytes(ca.public_bytes(serialization.Encoding.PEM))
    cert_pem = directory / "leaf.pem"
    cert_pem.write_bytes(leaf.public_bytes(serialization.Encoding.PEM))
    key_pem = directory / "leaf-key.pem"
    key_pem.write_bytes(leaf_key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL, serialization.NoEncryption(),
    ))
    return ca_pem, cert_pem, key_pem


def file_in_capath(ca_pem: pathlib.Path, directory: pathlib.Path) -> pathlib.Path:
    """Copy ``ca_pem`` into ``directory`` as ``<subject hash>.0``: OpenSSL's hash is the first four bytes, little-endian,
    of SHA-1 over the canonical subject without its outer SEQUENCE header, which for ``throwaway_ca``'s single lowercase
    UTF8String common name is the subject's own DER encoding."""
    from cryptography import x509

    der = x509.load_pem_x509_certificate(ca_pem.read_bytes()).subject.public_bytes()
    header = 2 if der[1] < 0x80 else 2 + (der[1] & 0x7F)
    digest = int.from_bytes(hashlib.sha1(der[header:]).digest()[:4], "little")
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{digest:08x}.0"
    target.write_bytes(ca_pem.read_bytes())
    return target


def trusted_certificate(ca_pem: pathlib.Path, *, reject: bool = False) -> bytes:
    """``ca_pem`` as a PEM ``TRUSTED CERTIFICATE``: its DER followed by OpenSSL's auxiliary trust (``X509_CERT_AUX``)
    naming serverAuth (1.3.6.1.5.5.7.3.1) as trusted, or as rejected (its ``[0]`` field), the certificate unchanged."""
    import base64
    import textwrap

    from cryptography import x509
    from cryptography.hazmat.primitives import serialization

    der = x509.load_pem_x509_certificate(ca_pem.read_bytes()).public_bytes(serialization.Encoding.DER)
    server_auth = bytes.fromhex("06082b06010505070301")
    aux = b"\x30\x0c" + (b"\xa0" if reject else b"\x30") + b"\x0a" + server_auth
    body = "\n".join(textwrap.wrap(base64.b64encode(der + aux).decode("ascii"), 64))
    return f"-----BEGIN TRUSTED CERTIFICATE-----\n{body}\n-----END TRUSTED CERTIFICATE-----\n".encode("ascii")


def throwaway_crl(directory: pathlib.Path) -> pathlib.Path:
    """A synthetic revocation list alone, not a certificate or a system trust change."""
    pytest.importorskip("cryptography", reason="cryptography is not installed")
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    now = _dt.datetime.now(_dt.timezone.utc)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "ouroboros throwaway test ca")])
    crl = (x509.CertificateRevocationListBuilder().issuer_name(issuer)
           .last_update(now - _dt.timedelta(days=1)).next_update(now + _dt.timedelta(days=1))
           .sign(key, hashes.SHA256()))
    directory.mkdir(parents=True, exist_ok=True)
    result = directory / "crl.pem"
    result.write_bytes(crl.public_bytes(serialization.Encoding.PEM))
    return result

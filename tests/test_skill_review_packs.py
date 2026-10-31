"""The reviewable skill payload: what a reviewer may see, and the payloads refused outright.

The binary policy judges file CONTENT, never a filename (#447 X4/В21): loadable native code
hard-blocks review by its magic bytes, other binaries and WebAssembly become typed descriptors,
and a text file survives whatever its extension; the one pasted rendering (``payload_pack``)
carries descriptors, never raw bytes, and is never split.
"""

from __future__ import annotations

import hashlib

import pytest

from ouroboros.skill_review_packs import (
    WASM_MAGIC,
    SkillBinaryPayload,
    _read_skill_file,
    executable_magic_kind,
    frozen_payload_files,
    payload_pack,
)


def _pack(skill_dir) -> str:
    return payload_pack(frozen_payload_files(skill_dir))


def test_non_magic_binary_becomes_a_typed_descriptor(tmp_path):
    """A non-UTF-8 blob that carries no loader magic is no hard block — raw bytes still never
    reach the reviewer, but the pack carries a typed {path,size,mime_from_name,sha256} descriptor
    so the reviewer can judge it on the merits instead of the whole skill going pending."""
    skill_dir = tmp_path / "skills" / "bin1"
    skill_dir.mkdir(parents=True)
    payload = b"\xff\xfeBEGIN CERT leak-me-please\xff\xc0\xc1\xfe\xff"
    (skill_dir / "cert.dat").write_bytes(payload)

    text, digest, descriptor = _read_skill_file(skill_dir / "cert.dat", relpath="cert.dat")
    assert text is None and digest
    assert descriptor is not None
    assert descriptor["path"] == "cert.dat"
    assert descriptor["size"] == len(payload)
    assert len(descriptor["sha256"]) == 64


def test_skill_review_blocks_loadable_native_binaries(tmp_path):
    """Loadable native code must hard-block review whatever its NAME says: the subprocess could
    otherwise ``ctypes.CDLL`` / import / require the blob and execute never-reviewed code."""
    skill_dir = tmp_path / "skills" / "nativelink"
    skill_dir.mkdir(parents=True)
    target = skill_dir / "evil.so"
    target.write_bytes(b"\x7fELF" + b"\x00" * 128)
    with pytest.raises(SkillBinaryPayload):
        _read_skill_file(target, relpath="evil.so")

    # A RENAMED ELF is still blocked: the magic bytes decide, not the suffix.
    renamed = skill_dir / "notes.txt"
    renamed.write_bytes(b"\x7fELF" + b"\x00" * 128)
    with pytest.raises(SkillBinaryPayload):
        _read_skill_file(renamed, relpath="notes.txt")


def test_skill_review_pack_carries_descriptor_not_raw_bytes(tmp_path):
    """The pasted rendering carries the JSON descriptor block for a non-UTF-8 file — never its raw
    bytes — while text files stay inlined; one pack, never split."""
    skill_dir = tmp_path / "skills" / "mixed"
    skill_dir.mkdir(parents=True)
    (skill_dir / "skill.json").write_text('{"name": "mixed"}', encoding="utf-8")
    payload = b"\x80\x81\x82opaque-data\xff"
    (skill_dir / "blob.dat").write_bytes(payload)

    joined = _pack(skill_dir)
    assert '{"name": "mixed"}' in joined
    assert "descriptor only" in joined
    assert hashlib.sha256(payload).hexdigest() in joined
    assert "opaque-data" not in joined  # raw bytes never inlined
    assert payload_pack([]) == "(empty skill directory — no manifest, no payload)"


def test_skill_review_blocks_executables_by_magic_not_filename(tmp_path):
    """Loadable executables are hard-blocked by CONTENT magic bytes (ELF/PE/Mach-O/.pyc)
    regardless of filename; a disguised extension does not evade the block."""
    import importlib.util

    skill_dir = tmp_path / "skills" / "nativelink2"
    skill_dir.mkdir(parents=True)
    samples = {
        "innocent.txt": b"\x7fELF" + b"\x00" * 128,          # ELF, disguised name
        "tool.dat": b"MZ\x90\x00" + b"\xff" * 64,            # PE (non-UTF-8 body)
        "lib.data": b"\xcf\xfa\xed\xfe" + b"\x00" * 32,      # Mach-O 64-bit LE
        "cache.dat": importlib.util.MAGIC_NUMBER + b"\x00" * 32,  # .pyc
    }
    for name, payload in samples.items():
        target = skill_dir / name
        target.write_bytes(payload)
        with pytest.raises(SkillBinaryPayload):
            _read_skill_file(target, relpath=name)


def test_skill_review_admits_wasm_as_content_hash_bound_descriptor(tmp_path):
    """WebAssembly is not a loader magic: the host never loads it natively (it runs only inside the
    browser's sandboxed widget frame), so a ``.wasm`` file takes the ordinary non-UTF-8 path — a typed
    descriptor the reviewer does not read — while the payload content hash binds every byte."""
    from ouroboros.skill_loader import compute_content_hash

    skill_dir = tmp_path / "skills" / "wasmpack"
    skill_dir.mkdir(parents=True)
    (skill_dir / "skill.json").write_text('{"name": "wasmpack"}', encoding="utf-8")
    wasm = b"\x00asm\x01\x00\x00\x00" + b"\x01\x85\x80\x80\x80\x00\xff\xfe" * 4
    (skill_dir / "core.wasm").write_bytes(wasm)

    assert executable_magic_kind(wasm, is_utf8_text=False) == ""
    text, digest, descriptor = _read_skill_file(skill_dir / "core.wasm", relpath="core.wasm")
    assert text is None and digest == hashlib.sha256(wasm).digest()
    assert descriptor is not None and descriptor.pop("mime_from_name")
    assert descriptor == {"path": "core.wasm", "size": len(wasm), "sha256": hashlib.sha256(wasm).hexdigest()}
    joined = _pack(skill_dir)
    assert "core.wasm (binary file — descriptor only" in joined
    assert hashlib.sha256(wasm).hexdigest() in joined
    before = compute_content_hash(skill_dir)
    (skill_dir / "core.wasm").write_bytes(wasm[:-1] + b"\x00")
    assert compute_content_hash(skill_dir) != before


def test_skill_review_routes_utf8_decodable_wasm_to_descriptor(tmp_path):
    """The descriptor route is chosen by the WebAssembly magic, not by a failed UTF-8 decode: the
    canonical 8-byte module and a functional ASCII-only module both decode as UTF-8, yet neither may
    be inlined as text. A native loader magic renamed ``.wasm`` still hard-blocks by content."""
    skill_dir = tmp_path / "skills" / "wasmtext"
    skill_dir.mkdir(parents=True)
    (skill_dir / "skill.json").write_text('{"name": "wasmtext"}', encoding="utf-8")
    canonical = WASM_MAGIC + b"\x01\x00\x00\x00"
    functional = (canonical + b"\x01\x05\x01\x60\x00\x01\x7f" + b"\x03\x02\x01\x00"
                  + b"\x07\x05\x01\x01f\x00\x00" + b"\x0a\x06\x01\x04\x00\x41\x2a\x0b")
    for name, module in (("empty.wasm", canonical), ("answer.wasm", functional)):
        module.decode("utf-8")  # the precondition the decode-first branch mistook for text
        (skill_dir / name).write_bytes(module)
        text, digest, descriptor = _read_skill_file(skill_dir / name, relpath=name)
        assert text is None and digest == hashlib.sha256(module).digest(), name
        assert descriptor is not None and descriptor["sha256"] == hashlib.sha256(module).hexdigest()
        assert descriptor["path"] == name and descriptor["size"] == len(module)
    joined = _pack(skill_dir)
    assert "empty.wasm (binary file — descriptor only" in joined
    assert "answer.wasm (binary file — descriptor only" in joined
    assert "\x00asm" not in joined  # module bytes never inlined, not even as "text"
    (skill_dir / "core.wasm").write_bytes(b"\x7fELF" + b"\x00" * 32)  # ELF disguised as wasm
    with pytest.raises(SkillBinaryPayload):
        _read_skill_file(skill_dir / "core.wasm", relpath="core.wasm")


def test_skill_review_does_not_block_text_by_scary_filename(tmp_path):
    """Capability preservation: the block is content-judged, so a valid UTF-8 text file survives
    review even with a formerly-blocking extension or an ambiguous 'MZ' text prefix."""
    skill_dir = tmp_path / "skills" / "textish"
    skill_dir.mkdir(parents=True)
    (skill_dir / "notes.so").write_text("just text, not an ELF", encoding="utf-8")
    text, _digest, descriptor = _read_skill_file(skill_dir / "notes.so", relpath="notes.so")
    assert text == "just text, not an ELF"
    assert descriptor is None

    (skill_dir / "mz.md").write_text("MZ stands for initials", encoding="utf-8")
    text, _digest, descriptor = _read_skill_file(skill_dir / "mz.md", relpath="mz.md")
    assert text == "MZ stands for initials"
    assert descriptor is None


def test_skill_review_retrieving_assignment_allows_binary_inspection():
    """A retrieving reviewer is told that a binary file is only its descriptor in the pack or index and
    that the frozen copy lets it inspect the file itself — a descriptor never claims its bytes were inspected."""
    from ouroboros.tools.review_skill import _PAYLOAD_INLINE, _PAYLOAD_SOURCES

    assert "{{path,size,mime_from_name,sha256}}" in _PAYLOAD_INLINE
    assert "{path,size,mime_from_name,sha256}" in _PAYLOAD_INLINE.format(content_hash="h", address="the copy")
    assert "never the mutable skill_dir path" in _PAYLOAD_INLINE and "never the mutable skill_dir path" in _PAYLOAD_SOURCES
    assert "say in the finding which you did" in _PAYLOAD_SOURCES


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_frozen_payload_files_carry_the_read_receipt_identity(tmp_path, newline):
    """Each text file of the frozen copy carries the identity a ``read_file`` receipt stamps, so a
    retrieving seat's coverage of it is observable; a binary file is only its descriptor."""
    skill_dir = tmp_path / "skills" / "ident"
    skill_dir.mkdir(parents=True)
    # Exact fixture bytes: Windows text-mode writes would translate LF to CRLF.
    payload = "---\nname: ident\n---\nbody\n".replace("\n", newline).encode("utf-8")
    (skill_dir / "SKILL.md").write_bytes(payload)
    (skill_dir / "blob.bin").write_bytes(b"\x80\x81")
    entries = {entry["path"]: entry for entry in frozen_payload_files(skill_dir)}
    assert entries["SKILL.md"]["sha256"] == hashlib.sha256(payload).hexdigest()
    assert entries["SKILL.md"]["source_revision"] == hashlib.sha256(payload).hexdigest()
    assert entries["SKILL.md"]["complete_sha256"] == hashlib.sha256(b"---\nname: ident\n---\nbody\n").hexdigest()
    assert entries["SKILL.md"]["text"] == payload.decode("utf-8")
    assert entries["SKILL.md"]["size"] == len(payload)
    assert entries["SKILL.md"]["complete_chars"] == len("---\nname: ident\n---\nbody\n")
    assert set(entries["SKILL.md"]) >= {"source_revision", "complete_sha256", "range_basis", "text", "size"}
    assert entries["blob.bin"]["binary"]["size"] == 2 and "text" not in entries["blob.bin"]

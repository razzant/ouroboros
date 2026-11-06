"""Existing binary review descriptors survive delegated payload capture/apply."""
import hashlib
import json
from pathlib import Path
import struct
import zlib


from ouroboros import delegate_custody as custody
from ouroboros.subagent_worktrees import find_execution_snapshot
from ouroboros.tools.delegate import _capture_terminal_patch
from ouroboros.tools.subagent_integration import _integrate_delegated_patch
from tests.test_delegated_skill_payload import _payload_entry, _provisioned



def test_binary_resource_capture_apply_preserves_exact_bytes_and_control_state(tmp_path, monkeypatch):
    ctx, skill, handle = _provisioned(tmp_path, monkeypatch)
    snapshot = Path(handle.path)
    def chunk(kind, body):
        return struct.pack('!I', len(body)) + kind + body + struct.pack('!I', zlib.crc32(kind + body))
    png = (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!2I5B', 1, 1, 8, 2, 0, 0, 0))
           + chunk(b'IDAT', zlib.compress(b'\0\xff\0\0')) + chunk(b'IEND', b''))
    (snapshot / 'image.png').write_bytes(png)
    state = ctx.drive_root / 'state/skills/alpha'
    state.mkdir(parents=True, exist_ok=True)
    grants = state / 'grants.json'
    grants.write_text('{"granted_keys":[]}\n')
    original_grants = grants.read_bytes()
    entry = _payload_entry(handle, skill)
    try:
        capture = _capture_terminal_patch(ctx, entry)
        assert capture['status'] == 'ready_with_changes', capture
        manifest = json.loads(Path(capture['manifest_artifact']).read_text())
        assert manifest['binary_files'] == [{'path': 'image.png', 'size': len(png), 'mime_from_name': 'image/png',
                                              'sha256': hashlib.sha256(png).hexdigest()}]
        result = _integrate_delegated_patch(ctx, 'run-p1', 'apply', 'preserve the reviewed image')
        assert '✅ Integrated' in result, result
        assert (skill / 'image.png').read_bytes() == png
        assert grants.read_bytes() == original_grants
        assert not (skill / '.git').exists()
        assert find_execution_snapshot('snapP') is None
    finally:
        custody._CUSTODY.clear()

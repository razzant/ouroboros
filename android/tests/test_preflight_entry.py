"""Exercise the Linux entry's real environment projection without root/mounts."""
import os
from pathlib import Path
import subprocess

import pytest


@pytest.mark.parametrize('override,expected', [(None, '3600'), ('5400', '5400')])
def test_android_entry_forwards_default_or_explicit_total_test_budget(tmp_path, override, expected):
    source = (Path(__file__).parents[1] / 'bootstrap/enter-linux').read_text()
    assignments = '\n'.join(line for line in source.splitlines() if line.startswith(('preflight_workers=', 'preflight_timeout=')))
    invocation = source[source.index('exec chroot '):]
    # Replace only the physical chroot boundary. Its real env -i argument list
    # remains intact, so a missing forward cannot pass from inherited state.
    shim = tmp_path / 'chroot'
    shim.write_text('#!/bin/sh\nshift\nexec "$@"\n')
    shim.chmod(0o755)
    environment = os.environ.copy()
    environment['PATH'] = str(tmp_path) + os.pathsep + environment.get('PATH', '')
    environment['OUROBOROS_PREFLIGHT_TEST_WORKERS'] = '3'
    environment.pop('OUROBOROS_PREFLIGHT_TIMEOUT_SEC', None)
    if override is not None:
        environment['OUROBOROS_PREFLIGHT_TIMEOUT_SEC'] = override
    result = subprocess.run(
        ['sh', '-c', 'set -eu\nroot=/unused\n' + assignments + '\n' + invocation,
         'entry-fixture', '/bin/sh', '-c',
         'printf \'%s\\n%s\\n\' "$OUROBOROS_PREFLIGHT_TIMEOUT_SEC" "$OUROBOROS_PREFLIGHT_TEST_WORKERS"'],
        env=environment, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [expected, '3']

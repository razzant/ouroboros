"""Runner-side subject binding. This module never imports or executes the checked out product."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

HERE = Path(__file__).absolute().parent
QA = '.github/p05-t3-qa'
WORKFLOW = '.github/workflows/p05-t3-qa.yml'
SOURCES = ('pins.json', 'p05_subject.py', 'p05_proof.py', 'run.sh', 'p05_live.py', 'p05_place_fixture.py',
           'test_p05_browser.py', 'test_p05_upgrade.py')
PUBLIC = 'https://github.com/razzant/ouroboros.git'
PY_FILE = re.compile(r'tests/(?:[a-zA-Z0-9_-]+/)*test_[a-zA-Z0-9_]+\.py\Z')
JS_FILE = re.compile(r'web/tests/(?:[a-zA-Z0-9_-]+/)*[a-zA-Z0-9_-]+\.test\.js\Z')
ENGINE_PIN = 'ouroboros/claudexor_runtime_pin.json'
# The bootstrap scripts and the runtime lock are byte-identical in every subject: one interpreter, one Node and one
# browser install serve all three. The project files and the engine pin are each subject's OWN trusted input: the
# older release keeps the lock it shipped with, and a candidate that moves dependencies keeps its own.
BOOTSTRAP_SHARED = ('requirements-runtime.lock', 'scripts/download_python_standalone.sh',
                     'scripts/ci_browser_install.py', 'scripts/download_node_standalone.py',
                     'scripts/download_node_standalone.sh')
BOOTSTRAP_PER_SUBJECT = ('pyproject.toml', 'uv.lock', ENGINE_PIN)
# An all-zero pin is the root's "not bound yet" marker: never a subject, so the wrapper cannot run unbound.
UNBOUND = '0' * 40
GIT_ENV = {'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8', 'GIT_CONFIG_GLOBAL': '/dev/null',
           'GIT_CONFIG_NOSYSTEM': '1', 'GIT_TERMINAL_PROMPT': '0', 'GIT_NO_REPLACE_OBJECTS': '1'}


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def sha(value, count=40):
    return isinstance(value, str) and re.fullmatch(rf'[0-9a-f]{{{count}}}', value) is not None


def git(path, *args):
    return subprocess.run(['/usr/bin/git', '--no-optional-locks', '-c', 'core.hooksPath=/dev/null',
                           '-C', str(path), *args], env=GIT_ENV, stdout=subprocess.PIPE, check=True).stdout


def obj(path, ref):
    return git(path, 'rev-parse', '--verify', ref).decode('ascii').strip()


def parents(path, ref):
    header = git(path, 'cat-file', 'commit', ref).split(b'\n\n', 1)[0]
    return [line[7:].decode('ascii') for line in header.splitlines() if line.startswith(b'parent ')]


def paths(output):
    return [value.decode('utf-8') for value in output.split(b'\x00') if value]


def regular_blob(repo, ref, path):
    expected = git(repo, 'ls-tree', '-z', ref, '--', path)
    require(expected.startswith((b'100644 blob ', b'100755 blob '))
            and expected.count(b'\x00') == 1 and expected.endswith(b'\t' + path.encode() + b'\x00'),
            f'Expected one regular source: {path}')
    return git(repo, 'cat-file', 'blob', f'{ref}:{path}')


def unbound(value):
    return isinstance(value, str) and set(value) == {'0'}


def load_pins(raw):
    value = json.loads(raw)
    require(value['format'] == 1 and set(value['subjects']) == {'base', 'candidate', 'old'}, 'Unsupported pin format')
    for role, subject in value['subjects'].items():
        require(set(subject) == {'commit', 'tree'} and all(sha(p) for p in subject.values()), f'Unset or invalid {role} pins')
    # Only the candidate may stay unbound, and only until the root binds its reviewed commit and tree.
    pending = [role for role, subject in value['subjects'].items()
               if unbound(subject['commit']) or unbound(subject['tree'])]
    require(not pending or pending == ['candidate'],
            'Only the candidate subject may be unbound, and it must be bound before a run')
    require(len({p['commit'] for p in value['subjects'].values()}) == 3, 'The three subjects must be separate')
    history = value['history']
    require(len(history) == 12 and len({r['commit'] for r in history}) == 12, 'Expected twelve distinct historical fixtures')
    for row in history:
        require(sha(row['commit']) and sha(row['tree']) and row['paths'], 'Invalid historical fixture pins')
        require(all(re.fullmatch(r'skills/[a-z_]+(?:/scripts/[a-z_]+\.py)?', path) and sha(oid)
                    for path, oid in row['paths'].items()), 'Unexpected historical fixture path or identity')
        if 'contents_sha256' in row:
            require(len(row['paths']) == 1 and sha(row['contents_sha256'], 64), 'Invalid historical blob digest')
    home = value['home']
    names = {'presence', 'queue', 'migration', 'telegram', 'history', 'structural', 'size', 'base_controls', 'node'}
    require(set(home) == names, 'Unexpected test matrix groups')
    for name, files in home.items():
        pattern = JS_FILE if name == 'node' else PY_FILE
        require(isinstance(files, list) and files and len(files) == len(set(files))
                and all(isinstance(f, str) and pattern.fullmatch(f) for f in files), f'Invalid home paths: {name}')
    selected = [path for name in names - {'base_controls'} for path in home[name]]
    require(len(selected) == len(set(selected)) and set(home['base_controls']) <= set(selected), 'Duplicate or unpaired controls')
    return value


def preflight(repo, event):
    require(sha(event) and obj(repo, 'HEAD') == event, 'Wrapper HEAD is not the exact event SHA')
    expected = {WORKFLOW, *(f'{QA}/{name}' for name in SOURCES)}
    for path in sorted(expected):
        regular_blob(repo, event, path)
    pins = load_pins(regular_blob(repo, event, f'{QA}/pins.json'))
    candidate = pins['subjects']['candidate']
    if unbound(candidate['commit']) or unbound(candidate['tree']):
        raise SystemExit('The candidate subject is not bound: set subjects.candidate.commit and subjects.candidate.tree '
                         'to the reviewed product commit and its tree before running this workflow.')
    prod, base = (pins['subjects'][key]['commit'] for key in ('candidate', 'base'))
    # The product commit sits directly on the exact public base (first parent); a second parent may carry a recorded merge.
    require(parents(repo, event) == [prod] and parents(repo, prod)[:1] == [base] and len(parents(repo, prod)) <= 2,
            'Expected a QA commit over a single reviewed product commit on the exact base')
    for role in ('candidate', 'base'):
        subject = pins['subjects'][role]
        require(obj(repo, subject['commit'] + '^{tree}') == subject['tree'], f'{role} tree does not match the pin')
    for filter_arg in ([], ['--diff-filter=A']):
        changed = paths(git(repo, 'diff', '--no-renames', '--name-only', '-z', *filter_arg, prod, event, '--'))
        require(set(changed) == expected and len(changed) == len(expected), 'Wrapper must only add the exact QA allowlist')
    selected = {file for group, files in pins['home'].items() if group != 'base_controls' for file in files}
    changes = paths(git(repo, 'diff', '--no-renames', '--name-only', '-z', base, prod, '--', 'tests', 'web/tests'))
    missing = [file for file in changes if (PY_FILE.fullmatch(file) or JS_FILE.fullmatch(file)) and file not in selected]
    # Do not silently omit an A/B/C module that moved or arrived after the initial matrix read.
    require(not missing, 'Changed home module missing from the pinned matrix: ' + ', '.join(repr(file) for file in missing))
    for file in sorted(selected):
        regular_blob(repo, prod, file)
    for file in pins['home']['base_controls']:
        require(regular_blob(repo, base, file) == regular_blob(repo, prod, file), f'Base control changed: {file}')
    return pins, expected


def verify_checkout(path, subject):
    meta = path / '.git'
    require(meta.is_dir() and not meta.is_symlink() and not (meta / 'objects/info/alternates').exists(), 'Expected a plain Git checkout')
    require(obj(path, 'HEAD') == subject['commit'] and obj(path, 'HEAD^{tree}') == subject['tree'], 'Checkout differs from pinned subject')
    config = git(path, 'config', '--local', '--name-only', '--list').decode('utf-8').lower().splitlines()
    require(not any(any(word in key for word in ('extraheader', 'credential', 'include', 'proxy', 'promisor'))
                    or key.startswith('url.') for key in config), 'Unexpected Git credential, redirect or partial clone config')
    require(not git(path, 'for-each-ref', 'refs/replace/') and not (meta / 'info/grafts').exists(), 'Unexpected Git history overlay')
    git(path, 'diff', '--quiet', 'HEAD', '--')
    require(not git(path, 'ls-files', '--others', '-z'), 'The execution checkout is not pristine')


def history(repo, fixtures):
    # Only the runner fetches public fixture objects, using no credential helper. Tests run without Git remotes.
    missing = [row['commit'] for row in fixtures if subprocess.run(
        ['/usr/bin/git', '-C', str(repo), 'cat-file', '-e', row['commit'] + '^{commit}'], env=GIT_ENV,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0]
    if missing:
        git(repo, '-c', 'credential.helper=', 'fetch', '--no-tags', '--depth=1', PUBLIC, *missing)
    for row in fixtures:
        commit = row['commit']
        require(obj(repo, commit + '^{tree}') == row['tree'], 'Historical root tree differs from its pin')
        for path, expected in row['paths'].items():
            require(obj(repo, f'{commit}:{path}') == expected, 'Historical path differs from its pin')
            if 'contents_sha256' in row:
                require(hashlib.sha256(regular_blob(repo, commit, path)).hexdigest() == row['contents_sha256'],
                        'Historical writer bytes differ from the fixture digest')


def node_dependencies(raw):
    """What scripts/download_node_standalone.py reads from the engine pin: release.node_version and every
    release.node_artifacts entry (archive_url, size_bytes, sha256, executable). The Claudexor release named beside
    them (version, build_sha, archive_url, sha256, size_bytes) never reaches the bootstrap."""
    release = json.loads(raw)['release']
    return release['node_version'], release['node_artifacts']


def stage(repo, root, sha_of_wrapper):
    pins, expected = preflight(repo, sha_of_wrapper)
    subjects = pins['subjects']
    for role, subject in subjects.items():
        checkout = repo / 'qa-sources' / role
        verify_checkout(checkout, subject)
        target = root / role
        shutil.copytree(checkout, target, symlinks=True)
        git(target, 'remote', 'remove', 'origin')
        require(not git(target, 'remote'), 'Unexpected remote in staged checkout')
        if role == 'candidate':
            require(obj(target, subjects['base']['commit'] + '^{tree}') == subjects['base']['tree'], 'Size comparison base is unavailable')
            history(target, pins['history'])
        verify_checkout(target, subject)
        require(git(target, 'write-tree').decode('ascii').strip() == subject['tree'], 'Index tree does not match HEAD')
    harness = root / 'harness'
    harness.mkdir()
    hashes = {}
    for path in sorted(expected):
        raw = regular_blob(repo, sha_of_wrapper, path)
        if path.startswith(QA + '/'):
            (harness / Path(path).name).write_bytes(raw)
        hashes[path] = hashlib.sha256(raw).hexdigest()
    # Each subject installs from its OWN trusted project files: the older release keeps the lock it shipped with and a
    # candidate that moves dependencies keeps its own. The scripts that fetch the interpreter, Node and the browsers are
    # byte-identical in all three subjects, and the Node dependencies the bootstrap reads from the engine pin must
    # agree, or one installed Node could not serve every subject. Copies here contain no candidate packages and no
    # importable Ouroboros runtime.
    bootstrap = root / 'bootstrap'
    shared = {}
    for path in BOOTSTRAP_SHARED:
        raw = regular_blob(root / 'base', subjects['base']['commit'], path)
        for role in ('candidate', 'old'):
            require(raw == regular_blob(root / role, subjects[role]['commit'], path),
                    f'Bootstrap script or runtime lock differs in {role}: {path}')
        shared[path] = raw
    node_deps = None
    for role in ('base', 'candidate', 'old'):
        target = bootstrap / role
        for path in BOOTSTRAP_SHARED:
            out = target / path
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(shared[path])
        for path in BOOTSTRAP_PER_SUBJECT:
            raw = regular_blob(root / role, subjects[role]['commit'], path)
            out = target / path
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(raw)
            if path == ENGINE_PIN:
                deps = node_dependencies(raw)
                require(node_deps is None or deps == node_deps, 'Bootstrap Node dependency changed: ' + path)
                node_deps = deps
    subject = {'format': 1, 'workflow_head': sha_of_wrapper, 'subjects': subjects,
               'harness_sha256': hashes, 'historical_fixtures': pins['history'], 'test_matrix': pins['home'],
               'bootstrap': {role: f'bootstrap/{role}' for role in ('base', 'candidate', 'old')}}
    (root / 'subject.json').write_text(json.dumps(subject, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def main():
    mode, source, event, *args = sys.argv[1:]
    repo = Path(source)
    if mode == 'verify' and len(args) == 1:
        pins, _ = preflight(repo, event)
        with Path(args[0]).open('a', encoding='utf-8') as output:
            for role in ('candidate', 'base', 'old'):
                print(f"{role}={pins['subjects'][role]['commit']}", file=output)
    elif mode == 'stage' and len(args) == 1:
        stage(repo, Path(args[0]), event)
    else:
        raise SystemExit('Expected verify REPO WRAPPER_SHA OUTPUT or stage REPO WRAPPER_SHA ROOT')


if __name__ == '__main__':
    main()

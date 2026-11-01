# Disposable distro bare-Python qualification

Runs the Unified Update's private-environment path on Ubuntu's own `/usr/bin/python3` in throwaway containers on a
GitHub-hosted runner. The cells have a package-owned `EXTERNALLY-MANAGED` marker, have `python3-venv` and pip really
absent where a cell says so, and run as root where a cell says so. Python's pip policy stays on throughout. Nothing here
sets `PIP_BREAK_SYSTEM_PACKAGES`, and `distro_overlay_plugin.py` removes the value the subject's `offline_pip` fixture
sets. The subject's test boundary is never removed.

| Cell | apt beyond `python3 git ca-certificates` | uid | pip path predicted | Beyond policy, path and census |
|---|---|---|---|---|
| `root-no-ensurepip-no-pip` | none | 0 | `pip_wheel`, the lock's pip fetched at its reviewed digest | real lock; real entry after selection and under a live handoff; no-network refusal |
| `user-own-pip-no-ensurepip` | `python3-pip` | non-root | `interpreter_pip` | toy entry, return and clone (tests 159, 198, 222) |
| `root-ensurepip` | `python3-pip python3-venv` | 0 | `ensurepip` | test 117 |

Three roles stay apart. The **target** is `/usr/bin/python3`, the frozen image's own. The **builder** is uv, which writes
test dependencies to a side directory with `--target`. The **driver** is pytest inside the target process. It sees that
directory through `sys.path` only, so every child runs on exactly what apt installed. A uv venv would make `sys.prefix`
a venv's, and the product reads it in-process.

`probe.py` measures each condition before any test. If the apt state is not what the cell exists for, the verdict is
`CONDITION_NOT_MET` and no product statement follows. `verdict.py` passes a cell only when every selected node
reported a pass, every census is clean and every phase container exited 0. A nonzero exit that no failing row accounts
for, a per-test timeout, or a census that could not read some bytes reads `INCOMPLETE`. Another legitimate pip path
than the predicted one reads `PATH_NOT_COVERED`.

To bind: fill `image.digest` in `manifest.json` and push once to `qualification/update-distro-bare-25fc93a8e3a4-r1`. To move to another
subject, `subject_check.py rebind manifest.json CHECKOUT REV` prints the new block and the changed consumers. The subject
commit must be reachable in the repository the workflow runs in, and the base commit's own workflows must not fire on
that branch (true at the pinned commit). Repeat attempts are refused; a rerun is a new push.

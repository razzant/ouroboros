# Real-data stand

The unit cost contract (`tests/test_cost_contract.py`) counts work on synthetic shapes. On 22.10 an update made one
install unusable (10–20 GB server memory, 77–155 % CPU, 4 of 5 direct turns never reached the model) while every
gate was green, because no gate ran on data shaped like that install's history. The stand answers the question the
unit gates cannot: does an ordinary owner message still reach its model on THIS install's real history while
background upkeep runs, for a chosen body (any commit)?

It boots the body as a real server on a contained copy of the install's data root, with a loopback stub model and no
Claudexor daemon, sends ordinary owner messages at fixed offsets after readiness, and records facts per message:
accepted, task received, first model dispatch, visible reply. It samples the server's memory, CPU and threads every
10 s and every thread's frames every 15 s (`probe/sitecustomize.py`). An unfinished turn is reported as censored,
never as zero. Nothing here judges a body; the report is facts for the person or the mind comparing arms.

## Protocol

Quit the app for the whole session. Cloning a running install's root (5.7 million files on 23.10) stalled its
supervisor loop three times for 2–9 minutes, during the snapshot and again during an arm's clone of the copy, so
both `snapshot` and `arm` refuse while the install's pid lock (`<home>/ouroboros.pid`) names a live process.

    python -m devtools.real_data.stand snapshot --live ~/Ouroboros/data --out STAND
    python -m devtools.real_data.stand arm --stand STAND --source <git clone> --rev <sha> --label G
    python -m devtools.real_data.stand arm --stand STAND --source <git clone> --rev <sha> --label C

Arms run one at a time on fresh clones of the same snapshot. A comparison names a reference arm (G, the last body the
install ran well) and the candidate (C); a candidate passes when its first-dispatch times and peak memory are no worse
than G's on the same snapshot. Defaults: sends at 30, 120, 240 and 360 s after readiness, a 420 s window, 180 s per
reply, a 24 GiB memory cap that censors the arm.

## Containment (`sandbox.py`, `snapshot.py`)

- The copy leaves out the engine data, process-ownership records, every credential in `settings.json` (model slots
  point at the stub, MCP and Telegram are off) and downloaded toolchains.
- SQLite stores are cloned without a lock and kept only when no commit or in-flight write overlapped the clone; hot
  journals are dropped, and each copy passes `PRAGMA quick_check`.
- The arm runs under a Seatbelt profile: writes only inside its own directory; no reads of the live stores, settings or
  engine data; no network except loopback, and not to the live app or its engine; no signals outside its sandbox; no
  `node`, so no Claudexor daemon. Fifteen canaries prove each rule from inside the profile before the server boots; a
  denial counts only as the sandbox's own refusal (EPERM), never as a missing file or closed port; any surprise stops
  the arm (`CanaryFailure`).

## Size and remover

Clones share blocks with their source, so a snapshot and each arm cost metadata plus what the arm writes (bounded by
the window and the memory cap; an arm prints its directory and free space before and after). Nothing is created
outside the stand directory. Delete a stand directory with a paced, cancellable delete when it is large.

macOS only (APFS `clonefile`, Seatbelt `sandbox-exec`).

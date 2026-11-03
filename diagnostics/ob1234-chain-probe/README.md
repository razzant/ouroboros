# Packaged Chromium chain observation (OB #1234)

This research-only method compares two published Linux tarballs, Ouroboros 7.4.4 and 7.9.0, in a disposable Silverblue 43 userspace container on a standard GitHub Ubuntu x86_64 runner. It measures a real packaged entry point, server, admitted task, Playwright driver and Chromium mappings. It is not a native Silverblue qualification or a repair of the reported SIGTRAP.

Only `run.sh`, `driver.py`, `observer.py`, `analyze.py`, `Containerfile`, this README and `sources.lock.json` belong in the published probe directory. `linux-appimage-native.yml` is a full research-branch-only replacement of the existing workflow with that path. Old proposals, handoffs, raw model requests, local paths and QA sources stay in private task evidence, not in this branch.

## One attempt

The first bounded run (41627217051) stopped before either arm: the pinned userspace manifest was unavailable. A separately authorized single repeat uses a freshly verified manifest and prepares Ubuntu's standard CNI plugins if absent. The earlier CNI warning did not establish the active network backend or cause the observed failure. Only the workflow on the reviewed research branch may be dispatched, once for this repeat, with its exact 40-hex `candidate_sha`. The original workflow body is a different experiment and must not be run for this investigation. The probe workflow has `workflow_dispatch` only, `contents: read`, SHA-pinned actions and `persist-credentials: false`. Its published logs contain only synthetic data, a fixture credential, disposable directories and official release metadata.

The 100-minute job bounds all steps. Arm steps are 19 minutes each; the driver's pre-teardown deadline is 700 seconds, outer exec 840 seconds plus a 30-second kill allowance, and the passive sampler TTL is 900 seconds. Stage errors remain errors. Collection, shutdown and artifact upload are attempted after earlier failures; runner loss or a job-level timeout can still prevent upload. There is no automatic retry of task submission or of the workflow.

## Source identity

`sources.lock.json` pins tarball URLs, sizes, SHA-256 and release commits. Each download must match both the pinned SHA and the release checksum file. The official Silverblue userspace image is digest-pinned. Tarballs are fetched sequentially and removed after verified extraction; both extracted payloads remain for the A/B run. Package metadata remains the original observation; the userspace pin has its own dated verification. Before downloading payloads, preparation reads that exact registry manifest with `skopeo inspect --raw`, preserving the response or error. Successful metadata reads cannot guarantee later blob availability.

SBOM equality of tar/deb/rpm is composition evidence, not proof of installer behavior. A versus B swaps the whole package: wrapper, launcher, libraries, Python and managed source. It is not a one-variable causal experiment. AppImage/FUSE/AppRun, rpm-ostree installation and an old launcher combined with updated source are not exercised. The manual arm C described in the private v2 proposal is not an enabled workflow lane.

## What happens

`run.sh` prepares a disposable directory on the runner, records host and disk facts, verifies payloads, builds the pinned userspace image and starts systemd/D-Bus in an ordinary container. It never requests privileged mode, disables seccomp/AppArmor/SELinux or changes the owner's machine. Container root is used for management and observation; the product runs as the dedicated `qa` user. Product-supplied Chromium flags, including its existing `--no-sandbox`, are not rewritten by the probe.

For each arm, `driver.py` runs the published entry point in a fresh HOME, configures a loopback scripted model through onboarding, waits for the `/api/state` supervisor/worker precondition, and submits one task. The actual POST response remains decisive: refusal or an unknown reply is saved without re-posting. The scripted provider offers `browse_page` and `browser_action` and advances only when its issued tool IDs return. Background/title/Safety requests cannot advance that sequence. The page exercises a simple canvas and a second-origin iframe. No separate renderer is promised; the product's site-isolation flags are unchanged.

Before fallible cleanup HTTP calls, the driver saves accumulated fixture facts. Later cleanup steps record independent outcomes. Shell cleanup captures remaining logs, stops the dedicated container user's processes, and ends the sampler with its recorded PID/birth identity. No extra Chromium `--version` process is launched after cleanup.

## Raw evidence and limits

The outside sampler records host PID, birth tick, namespace PID, parent birth when readable, executable, argv and selected loader/bus environment. Effective UID comes from `/proc/PID/status`, not the owner of its directory. Exe changes are re-identified and earlier images retained. Mappings are sampled only for identified Chromium processes. Failed identification, exe, environment or mapping reads remain visible. A missing/corrupt sampler file is unknown, not an empty successful observation.

Polling can miss short-lived processes and mappings; nested PID namespaces are outside this sampler. Parent links without a matching captured birth end as unknown. Disappearance is not an exit status. Only raw `DEBUG=pw:browser` lines can report the main browser's observed exit; this method cannot assert that every child avoided SIGTRAP.

`analyze.py` reads saved files only and emits facts, never a hypothesis verdict or PASS. Package-owner attribution is **not performed**: raw mapped paths are retained without a positional `rpm -qf` guess. Runtime browser version is unknown unless a real product/tool or debug line reports it; release metadata is labelled separately. One chosen largest log copy supplies each count (latest on ties), while all originals remain. Launcher and agent streams are counted separately. Generic Python processes remain role-unknown until a human joins actual worker/task events.

## Disk

The 3 GB preservation reserve (`OB1234_RESERVE_BYTES`) is an authored operational choice, not a measured peak or guarantee. Available space is measured before downloads, image creation, container start and each arm. Known download sizes are added to the reserve; unknown expanded sizes are not invented. The recorded payload/HOME sizes include managed repositories and dependency caches. Checks happen at stage boundaries, so a stage can still exhaust disk. No pre-existing runner data is deleted to make the result pass. A preparation failure is a useful recorded obstacle, not a successful browser experiment.

## What a result can establish

A successful task and readable mappings provide a new observation of the published entry-point-to-Chromium path and which library paths were actually seen. A failed preparation, admission, launch or read supplies its exact remaining obstacle. Neither outcome proves the historical native Silverblue 43 crash fixed: the runner has a different kernel, no enforcing Silverblue SELinux policy, no reporter installation/userbase, no desktop seat or physical GPU, and no original crash input or core.

The parent independently checked readiness fields against both exact Git tags and tested synthetic failure paths. Those local checks are not a Linux execution. Their source-bound receipts remain with the private task; the public run must be judged from its own raw artifact.

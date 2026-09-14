# Android backup, recovery and removal

This advanced guide accompanies [Android installation](ANDROID_INSTALL.md).
Commands use a POSIX shell on macOS or Linux, authorized ADB and Magisk root.
Windows shell equivalents and migration to another phone or ROM are unqualified.
No firmware flashing or bootloader operation is involved.

**Qualification:** a local Pixel test created an 11,820,506,112-byte runtime
archive and exported the personally signed host APK. All transfers completed and
the full archive checksum matched. Archive inspection found no captured `/proc` or
`sysfs` contents; nine critical file hashes matched, and the extracted Git data
passed `git fsck --full`, retaining both repository HEAD and the saved work stash.
Extraction into a separate phone directory passed critical content and metadata
comparisons. Activating that restored copy and removing the live installation
have not been qualified. A separate Magisk BusyBox 1.36.1.1 fixture preserved
file/directory modes, numeric ownership, hardlinks, symlink text and file contents
with the tar commands used below.

## What the backup contains

`runtime.tar` contains the entire `/data/local/ouroboros-phone` directory:

- Linux dependencies, bootstrap tools and the immutable launcher seed;
- complete source repository, Git history and uncommitted files;
- settings, memory, task history, artifacts and projects inside the installation;
- the permanent APK signing key **and its password file**;
- Linux home directories, harness state, download caches and retained builds.

`Ouroboros.apk` is a separate copy of the installed, personally signed Android
host. The publisher-signed reference APK is a different signing identity.

This is **not an Android app-data or OS backup**. It excludes Android app/WebView
storage, package-manager state, Android permissions, document-access grants,
Magisk authorization and files outside the installation, including `/sdcard`
outputs. Back those up separately where needed. The host declares
`allowBackup=false`; standard Android app backup does not replace this procedure.
Restoring harness files also does not guarantee that vendor sessions or credentials
remain valid on another device.

Treat the archive as private: it contains credentials, conversations and the
signing identity. Do not upload it to an issue, public release or review packet.
Keep a copy off the phone. This tar procedure preserves ordinary Unix metadata;
ACLs, extended attributes and SELinux labels for arbitrary customized software
are not a qualified migration contract. Unix-domain sockets are temporary IPC
endpoints and are not archived. The local test omitted 14 stale sockets under
Linux `/tmp`; they were not user documents or running processes.

## Stop before copying

Finish active work, then use **Panic / Остановить агента**. Wait for the core,
launcher and owned Claudexor work to stop; investigate any unconfirmed stop.
Force-stopping the APK alone does not stop Linux. Panic leaves background work
stopped until you explicitly resume it.

Set the device and a fresh backup name in the computer terminal:

```sh
set -eu
umask 077
ANDROID_SERIAL='YOUR_ADB_SERIAL'
recovery_tag=$(date -u +%Y%m%dT%H%M%SZ)
backup_dir="$HOME/ouroboros-backups/$recovery_tag"
mkdir -p "$backup_dir"
adb -s "$ANDROID_SERIAL" shell -T su -c 'am force-stop ai.ouroboros.android'
```

The script below checks the Panic marker and refuses to copy while a process
remains chrooted into the installation or a mount beneath it is visible. It does
not kill processes. Leave room on the phone for an additional uncompressed archive
and, for the restore check, an extracted copy. The measured archive size above
includes caches and builds; it is not a universal disk requirement.

## Create the archive and export the APK

```sh
adb -s "$ANDROID_SERIAL" shell -T su -c "sh -s -- '$recovery_tag'" <<'PHONE'
set -eu
umask 077
base=/data/local/ouroboros-phone
out=/data/local/tmp/ouroboros-recovery-$1
bb=/data/adb/magisk/busybox
test "$(cat "$base/rootfs/opt/ouroboros/data/state/panic_stop.flag")" = panic
for process in /proc/[0-9]*; do
    process_root=$(readlink "$process/root" 2>/dev/null || true)
    if [ "$process_root" = "$base/rootfs" ]; then
        echo "Runtime process is still alive: ${process##*/}; finish stopping it." >&2
        exit 1
    fi
done
awk -v prefix="$base/" 'index($5, prefix) == 1 { found=1 } END { exit found }' /proc/self/mountinfo
test ! -e "$out"
mkdir -m 700 "$out"
cd "$base"
test -s rootfs/opt/ouroboros/signing/host.keystore
test -s rootfs/opt/ouroboros/signing/host-password
for name in signing/host.keystore signing/host-password data/settings.json \
    data/memory/identity.md data/memory/scratchpad.md repo/VERSION repo/.git/HEAD; do
    path=rootfs/opt/ouroboros/$name
    if [ -f "$path" ]; then sha256sum "$path"; fi
done > "$out/critical.sha256"
"$bb" stat -c '%a:%u:%g:%F:%n' rootfs/opt/ouroboros/signing/host.keystore \
    rootfs/opt/ouroboros/signing/host-password rootfs/opt/ouroboros/launcher/current \
    > "$out/critical.metadata"
readlink rootfs/opt/ouroboros/launcher/current >> "$out/critical.metadata"
"$bb" tar -cf "$out/runtime.tar" -C /data/local ouroboros-phone
package_path=$(pm path ai.ouroboros.android)
case "$package_path" in package:*) package_path=${package_path#package:} ;; *) exit 1 ;; esac
cp "$package_path" "$out/Ouroboros.apk"
cd "$out"
sha256sum runtime.tar Ouroboros.apk > SHA256SUMS
chmod 600 runtime.tar Ouroboros.apk SHA256SUMS critical.sha256 critical.metadata
PHONE
```

Each command must finish successfully. Keep errors from an incomplete attempt;
resolve them and use a new backup name rather than assuming a partial archive is
usable. The APK export covers this host's single APK, not arbitrary split packages.
Do not delete the original installation or phone export yet.

## Transfer and verify on the computer

These reads use `adb shell -T`, which preserves the remote shell's failure status,
and do not make the root-owned backup world-readable:

```sh
for name in runtime.tar Ouroboros.apk SHA256SUMS critical.sha256 critical.metadata; do
    adb -s "$ANDROID_SERIAL" shell -T su -c "cat '/data/local/tmp/ouroboros-recovery-$recovery_tag/$name'" > "$backup_dir/$name"
done
python3 - "$backup_dir" <<'PYTHON'
import hashlib, pathlib, sys, tarfile
root = pathlib.Path(sys.argv[1])
for line in (root / 'SHA256SUMS').read_text().splitlines():
    expected, name = line.split(None, 1)
    path = root / name.strip().lstrip('*')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise SystemExit('Checksum mismatch: ' + path.name)
with tarfile.open(root / 'runtime.tar') as archive:
    for line in (root / 'critical.sha256').read_text().splitlines():
        expected, name = line.split(None, 1)
        with archive.extractfile('ouroboros-phone/' + name.strip().lstrip('*')) as stream:
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise SystemExit('Critical archive file does not match its source.')
print('Archive/APK transfer and recorded critical file hashes match.')
PYTHON
```

Keep the verified backup in private storage on another device. Checksums detect
changes; they do not make a backup from an unknown source trustworthy. Do not
extract the Linux tree onto macOS/Windows and treat that as a faithful Linux restore.

## Restore into a separate directory first

Keep the original installation stopped during comparisons. The following command
extracts beside it and starts nothing. Do not run the staged copy's bootstrap:
normal launch paths intentionally refer to the canonical installation.

```sh
adb -s "$ANDROID_SERIAL" shell -T su -c "sh -s -- '$recovery_tag'" <<'PHONE'
set -eu
umask 077
source=/data/local/tmp/ouroboros-recovery-$1
stage=/data/local/ouroboros-restore-check-$1
bb=/data/adb/magisk/busybox
test ! -e "$stage"
mkdir -m 700 "$stage"
"$bb" tar -xpf "$source/runtime.tar" -C "$stage"
cd "$stage/ouroboros-phone"
sha256sum -c "$source/critical.sha256"
"$bb" stat -c '%a:%u:%g:%F:%n' rootfs/opt/ouroboros/signing/host.keystore \
    rootfs/opt/ouroboros/signing/host-password rootfs/opt/ouroboros/launcher/current \
    > "$source/restored-critical.metadata"
readlink rootfs/opt/ouroboros/launcher/current >> "$source/restored-critical.metadata"
"$bb" cmp "$source/critical.metadata" "$source/restored-critical.metadata"
echo "Restored content and recorded metadata match; nothing was started."
PHONE
```

Inspect the restored source history and your important project/task files before
activation. These checks cover transfer, extraction and the recorded critical
files, not every application's semantic state or boot behavior. For a later restore,
set `recovery_tag` and `backup_dir` to the saved backup. If the phone export is gone,
copy the verified files back into a new private staging directory and verify their
SHA-256 there before extracting. Avoid shared `/sdcard` storage for private backups.

## Activate a verified backup

**Activation remains unqualified.** This is a manual same-phone recovery procedure,
not an automatic rollback or a promise of compatibility with another ROM.
Stop and verify the current runtime as above. Preserve it by rename instead of
unpacking over it:

```sh
adb -s "$ANDROID_SERIAL" shell -T su -c "sh -s -- '$recovery_tag'" <<'PHONE'
set -eu
base=/data/local/ouroboros-phone
stage=/data/local/ouroboros-restore-check-$1/ouroboros-phone
previous=/data/local/ouroboros-phone.before-restore-$1
test -d "$stage/rootfs/opt/ouroboros/repo"
test ! -e "$previous"
if [ -e "$base" ]; then mv "$base" "$previous"; fi
mv "$stage" "$base"
PHONE
```

Keep the installed personally signed APK if it belongs to this installation.
Do not install an older APK over a newer version, or uninstall to bypass a signer
mismatch. The restored source's ordinary Start/update hook can rebuild with the
restored personal key at a higher local versionCode. If the host package is absent,
install the saved personal APK:

```sh
adb -s "$ANDROID_SERIAL" install "$backup_dir/Ouroboros.apk"
```

Open Ouroboros and explicitly Start. Regrant Android/Magisk permissions if needed;
verify the core, bridge, source/version, personal data and a harmless task. The
backup's Panic marker should prevent automatic resumption until that explicit
Start. Keep the previous directory until recovery is proven. Restoring Linux
files does not restore Android app data or guarantee provider authentication.

## Remove the installation

**Live removal remains unqualified.** This deletes data; keep a verified off-phone
backup first. Finish work, Panic, force-stop the APK and verify Linux is stopped.
Remove the package:

```sh
adb -s "$ANDROID_SERIAL" uninstall ai.ouroboros.android
```

Android app data and grants are removed, but the Linux directory remains. Rename
that exact directory before deleting it so you can still inspect or recover it:

```sh
adb -s "$ANDROID_SERIAL" shell -T su -c "test ! -e '/data/local/ouroboros-phone.to-remove-$recovery_tag' && mv /data/local/ouroboros-phone '/data/local/ouroboros-phone.to-remove-$recovery_tag'"
```

After checking the backup and the exact directory selected for deletion:

```sh
adb -s "$ANDROID_SERIAL" shell -T su -c "rm -rf '/data/local/ouroboros-phone.to-remove-$recovery_tag'"
```

This removes the runtime, personal data and signing key still inside that
renamed directory. It leaves separately named backups, `/sdcard` outputs,
Magisk, firmware and unrelated apps untouched. The installer creates no Magisk
boot module or service script to remove; its boot entry belongs to the APK.
Older manually customized installations may have their own extra components.

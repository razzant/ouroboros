# Ouroboros on Android

**Experimental, under qualification.** The USB installer and phone runtime have
been exercised on a Pixel 10a using local, unpublished builds, including an
interrupted installation resumed after a fix. Short unplugged and forced-Doze
checks have passed; battery endurance, boot before first unlock, the complete
self-modification cycle and final attested release artifact remain under
qualification. This page does not certify another device.

Ouroboros runs on the phone: the ordinary core, source checkout, tools, memory,
projects and launcher all live there. Language-model inference uses your API or
subscription accounts. The Android host supplies the interface and native Android
access. A computer is needed for the initial USB setup; afterward the phone uses
its own internet connection.

## Requirements

- An ARM64 Android phone with working **Magisk root**, a functioning WebView and
  internet access. The installation uses Magisk BusyBox, private mounts and a
  Linux chroot. Other root implementations have not been qualified.
- A computer with Python 3.10 or newer, Android Platform-Tools (`adb`) and the
  GitHub CLI (`gh`) for public artifact attestation checks, a USB data cable,
  USB debugging enabled, and the computer authorized on the phone. See the
  [official ADB instructions](https://developer.android.com/tools/adb).
- Your own supported model-provider API credentials or compatible subscription
  account, configured through the ordinary Ouroboros setup.
- Free space for upstream downloads, the Linux environment and retained build
  artifacts. One local runtime backup, including caches and build artifacts,
  occupied about 11.8 GB uncompressed; this is an observation, not a minimum-space
  requirement. A supported memory minimum and battery endurance remain unqualified.

The manifest's minimum API value is a build declaration, not proof that every
Android 8 or newer device works. Pixel 10a is the tested example; the software
does not use the device brand as an admission rule.

This guide starts **after rooting**. Follow the
[official Magisk installation guide](https://topjohnwu.github.io/Magisk/install.html)
and your device vendor's instructions for your exact firmware. Bootloader unlocking
can erase the phone; preserve your data first. An Ouroboros release contains no
patched boot image and performs no firmware flashing.

## Choose and verify the release

Use an Android-capable release in the
[common Ouroboros release list](https://github.com/razzant/ouroboros/releases).
Keep all downloaded files from the same exact release tag.

| File | Purpose |
| --- | --- |
| `Ouroboros-{version}-android-arm64.tar.gz` | USB installer, Android sources and bootstrap, plus the common Git repository bundle and source manifests. |
| `Ouroboros-{version}-android.apk` | Publisher-signed reference host APK. Installing it alone does not create the Linux runtime and is not the personal-key setup procedure. |
| `SHA256SUMS`, `release-evidence.json`, `release-smoke-*.json`, `sbom-*.cdx.json` | Exact artifact hashes, source/workflow binding, artifact checks and component inventories. These are evidence files, not extra installers. |

Verify the archive against that release's `SHA256SUMS` and use the provenance
verification commands in its release notes. The archive's
`android_release_manifest.json` binds its source files and the reference APK to
the recorded source commit and release tag. CI checks package contents, signing
and source identity; it does not prove root, boot, hardware or battery behavior.

The archive does not contain a populated phone filesystem, SDK, provider
credentials, memory or private signing key. Installation obtains pinned inputs
from their upstream sources and keeps a durable download cache on the computer.
Internet access is required for the first setup; retain the cache for retries.

Ubuntu packages come from a dated HTTPS snapshot with Ubuntu's signed repository
metadata. A fresh Ubuntu Base has no CA package yet, so the installer uses the
phone's existing Android system CA certificates for initial HTTPS access; Ubuntu's
`ca-certificates` package then manages the Linux bundle normally. TLS certificate
checks and Ubuntu repository signature checks remain enabled.
Read and accept the upstream licenses requested by the installer, including the
[Android SDK terms](https://developer.android.com/studio/terms). Third-party
components retain their own licenses; Ouroboros's MIT license does not replace
them. Installed AAPT2/zipalign sources retain `LICENSE.txt` and vendor `NOTICE`
files under `/opt/ouroboros/toolchain-source/android-build-tools`; installed Ubuntu
packages keep their copyright files under `/usr/share/doc/` inside the rootfs.
The installer also retains the original upstream downloads in its cache, including
the Google SDK ZIPs: the installed SDK directory contains selected compiler files,
not a consolidated copy of every upstream notice. Preserve these source/archive
notices when maintaining or redistributing components. SDK, rootfs and browser
dependencies are downloaded upstream, not bundled in the Ouroboros source release.
This first experimental channel is GitHub, without a Google Play or offline bundle
distribution promise.

## Install through USB

Extract the archive and connect the already rooted phone. Keep the phone unlocked
for USB authorization, Magisk and Android permission dialogs.

From the extracted `Ouroboros-Android` directory, inspect prerequisites first.
Replace the archive path and `VERSION` with the file you downloaded:

```sh
python3 android/install.py --archive /path/to/Ouroboros-VERSION-android-arm64.tar.gz --check
```

This verifies the source archive and reads the attached phone's prerequisites;
it does not install the runtime. `--help` is offline. Public archives use GitHub
attestation verification before device inspection. If the computer is not yet
authorized for ADB or the USB shell lacks Magisk root, approve the corresponding
on-phone request and repeat the check.

After reading and accepting the SDK terms, install with:

```sh
python3 android/install.py --archive /path/to/Ouroboros-VERSION-android-arm64.tar.gz --accept-sdk-license
```

Add `--serial ADB_SERIAL` to select a particular authorized device; otherwise
exactly one authorized device must be connected. The default durable cache is
`~/.cache/ouroboros/android`; use `--cache-dir /path/to/cache` to move it. DNS is
read from Android's active network. If that discovery fails, `--dns ADDRESS`
supplies a resolver reachable through the phone's network; repeat the option
for more than one address. Do not copy another installation's router or VPN DNS.

While the native service runs, it refreshes Linux `resolv.conf` when Android's
default-network DNS changes. A missing network or empty DNS list keeps the last
file and shows a warning, rather than claiming connectivity. The callback sees
the Android app's default network; Linux runs as root (UID 0), so VPN routing can
differ. This refresh does not change VPN configuration. Linux's plain DNS resolver
does not implement Android's [Private DNS/DNS-over-TLS policy](https://developer.android.com/reference/android/net/LinkProperties#isPrivateDnsActive());
that limitation is reported when Private DNS is active. Verify actual Linux DNS
and HTTPS access after a network change. DNS refresh neither starts the core nor
resumes work stopped with Panic.

If setup must use an existing proxy reachable from the phone, add `--proxy http://HOST:PORT`.
This proxy applies only to the installation commands inside Linux; it is not saved in
Ouroboros settings or later runtime startup. Local browser-install traffic bypasses it.
The installer does not create a proxy server or change Wi-Fi/VPN settings. A warning
that Android has no active default network is informational: an explicit USB/proxy
route can still work, but a VPN icon or listed DNS servers alone do not prove internet
access. Package and provisioning output is shown live and retained under
`<cache-dir>/install-logs/<attempt>/`, with each command's exit code beside its log.
Independent operation after setup still requires the phone's own working internet.

For an explicitly trusted **unpublished test artifact** only,
`--expected-sha256 DIGEST` selects owner-supplied digest verification instead of
GitHub attestation. The result records `owner_digest`; this is not a claim of
publisher provenance and is not the normal release-install path.

The setup provisions a clean Linux environment, installs the shared
source and launcher, creates the installation's permanent private signing key,
then builds and installs the host from that source. It reports each completed
stage. If installation is interrupted, run the same command with the original
archive and cache; verified downloads are reused and provisioning resumes. A
completed installation is detected and retains its key, memory, settings and
source history, directing you to ordinary Updates. An existing unmanaged rootfs
is refused rather than overwritten. Local Pixel qualification resumed installation
successfully with the same personal key; final release-artifact qualification
remains pending. This command is not a reset operation.

The fixed device layout is `/data/local/ouroboros-phone`. Inside its `rootfs`,
`/opt/ouroboros` contains the source repository, data, tools and signing material.
This is phone storage; no directory on the setup computer is needed for normal
operation afterward.

Open Ouroboros and allow its Magisk root request if you want the selected broad
device access. Complete the ordinary provider/account setup with your own
credentials. The native **⋮** menu currently labels Start as **Запустить ядро**,
Status as **Статус ядра**, and Android access setup as **Разрешения Android**.

## Access and permissions

### Choosing Ouroboros as the Android assistant

On Android 10 and later, open the Ouroboros menu and choose **Назначить
ассистентом Android**. The app uses Android's normal `RoleManager` consent
screen, then reads the role back. Root-only `cmd role add-role-holder` is useful
for qualification scripts, but it is not the public user flow. The APK also
declares `ACTION_ASSIST`, which makes the existing Activity a candidate.

This entry opens the ordinary Ouroboros Activity. It does not provide a
`VoiceInteractionService`, hotword, assist context, or a guarantee that the full
WebView is available over a locked keyguard. The core and other background work
can continue after the phone has been unlocked once; a cold reboot before the
first unlock is a separate Direct Boot case and is not promised by this APK.
The rooted Linux tree lives under `/data/local/ouroboros-phone`, so its files
are physically present before unlock; the startup policy still waits for the
normal post-unlock Android boot event before starting the credential-dependent
core. The Direct Boot receiver records the locked-boot marker in Android's
device-protected app storage, but that marker does not make the Linux tree
credential-encrypted or start the core before unlock.

### Optional Android control surfaces

The APK declares four opt-in surfaces: Accessibility for owner-requested UI
actions, Notification Listener for notification readback, a live wallpaper, and
a Quick Settings tile. Their declaration is not consent. The public setup path
must use Android's Settings consent screens and read the enabled state back;
private rooted qualification may enable them through the owner-authorized fast
path. The bridge reports typed `accessibility.*` and `notifications.*` state so
the agent can discover whether a capability is actually available instead of
assuming it from the APK manifest.

Three independent permissions are involved:

- **Magisk root access** lets Ouroboros operate broadly on the phone through its
  root-side tools.
- **Android app permissions** govern native app-UID access: reading or changing
  contacts/calendar entries, reading permitted photos/video/audio, using the
  camera/microphone, and requesting approximate or precise location. First-open
  access setup lets you choose those grants in Android dialogs; **Later (Позже)** defers
  setup and the native menu reopens it. Android settings retain denial, revocation
  and available limited-access choices. Root-side tools retain their separate,
  broader authority.
- On Android 10 and later, unattended location needs the separate
  **Allow all the time** grant. After granting it, start or refresh native
  status so its foreground-service type includes location. Without that grant,
  Android may restrict background requests; foreground queries remain available.
- **Provider/account authorization** permits model and service access through
  the accounts you configure. Neither root nor an APK certificate supplies it.

`location.get` reports the location's timestamp, age and mock-provider flag so a
cached or test fix is distinguishable from a hardware observation. A timed-out
current request reports `no_fix_within_timeout`; a completed request without a
fix reports `provider_returned_null` and includes the background-grant state.
Neither result diagnoses the sensor or platform policy. Older last-known-only
reads can report `no_last_known_fix`. An SDK test-provider fix exercises callback
delivery but does not qualify GPS reception on a physical device.

In Settings → Available subagents, each coding session has its own access choice:
**Working files** (`workspace_write`, the default) or **Full system access** (`full`).
The tested Android kernel cannot create Codex's native sandbox, so a Codex coding
session needs Full system access. Claude Code under root rejects its full-access
mode; keep its working-files choice. This is saved session configuration, not an
automatic change of harness or permissions. Working changes still live in a private
snapshot until Ouroboros reviews and applies them; that snapshot is not a filesystem
sandbox. The host creates an initial full-access grant only when the selected
project has no trust record; an existing record that disables full access is
preserved. Existing sessions on every platform keep
Working files unless you change them. Read-only assignments remain read-only;
Codex's readonly sandbox is not qualified on the tested Android kernel.

The bridge also provides a typed `packages.install` path for an APK exposed by a
`content://` or other readable URI. It stages bytes through Android's
`PackageInstaller`, records the source URI, size and SHA-256, and requires a caller
supplied idempotency key. The call is asynchronous: `submitted` means that staging
and commit submission were observed, while `packages.install.status` and
`packages.sessions` are required to observe completion. A lost response is unknown
and is never retried automatically; a duplicate key with the same digest returns
the original receipt. Android may still return `pending_user_action` when the
installer policy requires owner consent. The adapter does not claim rollback
support, so an update caller must preserve the prior APK and verify the installed
package before deciding whether to recover. A pending install posts an Ouroboros
notification that opens Android's original confirmation screen when tapped;
it remains incomplete until the system reports success or failure. Keep
Ouroboros notifications enabled for this handoff. A disabled or failed notification
is reported in `confirmation_delivery`; it is not successful consent. Use
`packages.install.abandon` with the same idempotency key to close a rejected or
unknown installer session after inspecting it; it records an abandoned terminal
outcome and never starts a replacement installation.
For this path, enable **Install unknown apps** for Ouroboros in Android settings;
`capabilities.can_request_package_installs` reads that special access separately
from ordinary runtime permissions. The host can read only source URIs available
to its app UID. A Linux download can be written to a writable Android provider
URI with `content.write` and installed from that URI; a root-readable chroot path
does not itself grant the Android app access.

The manifest intentionally declares `QUERY_ALL_PACKAGES` for general installed-app
discovery and component inspection through Android's PackageManager. This is a
[normal install-time permission](https://developer.android.com/reference/android/Manifest.permission#QUERY_ALL_PACKAGES),
so Android shows no runtime grant dialog for package visibility. Visibility itself
does not grant access to another app's private contents. This GitHub experimental
distribution makes no claim of Google Play policy approval for broad visibility.

When the agent reads app data through native or root-side tools, returned content
can enter its task context, be sent to your configured model/API or subscription
provider, and remain in local tool history. Phone-resident execution does not mean
that model inputs remain on the phone. The bridge answers requested operations;
declaring a permission alone does not upload a contacts, media or location dataset.

An allowed root request does not prove that an Android content provider or a
WebView media request has its required app permission. A permission grant also
does not prove a camera frame or location fix was obtained. Android may request
new permissions after an update or revoke existing access; normal operations use
the current grant state rather than asking you to approve every agent action.

The interface connects to HTTP on **phone loopback** (`localhost:8765`), without
a computer-hosted core or public tunnel. The host denies cleartext HTTP by
default and permits it only for the loopback names used by the local core. This
keeps the local transport working while preventing an accidental external HTTP
request from being silently accepted by the WebView. The separate Linux runtime
still has its own network policy, and VPN/DNS remain your configuration choices.

## Check operation without the computer

After setup, run a harmless task and confirm its actual result and saved history.
Then disconnect USB and repeat through the phone's own connection. Check the
native Status view separately from the core interface: a running core and an
available Android bridge are distinct facts.

Before relying on background operation, exercise app close/reopen, core Restart,
reboot followed by first unlock, and screen-off operation both on charge and on
battery. A foreground notification is not a guarantee of uninterrupted work on
every Android build. The release's verification record should identify the exact
device, Android build and observed cases. Local Pixel tests have exercised
first-run Restart (exit 42), Panic stopping the core and launcher (exit 99), and
recovery after core process loss. During a physical USB disconnection, five
phone-side samples over 20 seconds confirmed battery operation, the same core and
launcher, and working core health and native bridge. A separate 15-second forced
Doze check observed deep idle with the screen off and three successful health and
bridge checks. These short checks do not establish battery endurance or sustained
network/model work in Doze. Boot before first unlock remains under qualification.

Live camera frames and microphone tracks were observed in the actual host WebView.
Closing the Activity and terminating its renderer released the native camera and
microphone operations before test cleanup. Revoking either permission during
capture caused Android to terminate the host and end both native operations;
the core survived and the original grants were restored. JavaScript track state
after host-process death was not inspectable. Allow/deny paths were exercised;
fresh location fixes and provider recovery remain unqualified.

The rooted Linux entry retains root while restoring ordinary OOM selection. Full
test preflight uses two workers and a one-hour total budget by default, preserving
explicit operator overrides. Targeted Python 3.12 and root test regressions were
fixed and checked on the phone, including an unbounded mock response that inflated
test memory. A complete passing preflight and self-edit → review → commit → restart
proof remain required; those focused checks do not substitute for them.

Panic is a full stop; the native **Остановить агента** action invokes it too.
Automatic entry must keep it stopped; use the explicit Start action when you want
to resume. Reopening the interface or rebooting is not permission to silently
resume stopped work.

## Updates and the personal signing key

Official core changes arrive through the existing Ouroboros update flow. Local
self-changes remain in the same source history and use the existing reviewed
merge/recovery mechanisms. Unchanged host and platform inputs reuse the installed APK. When
host sources, the icon, SDK installation or common `VERSION` change,
`android/bootstrap/update-host` builds the selected current
source and replaces the installed host using **the same personal key** and a
higher local `versionCode`. It checks the installed certificate and APK bytes
before reporting success; a missing key stops the update rather than generating
a different identity.

Updates also apply changed Android SDK/compiler inputs and patches, the common
Node toolchain, browser inputs and the supported Linux package recipe from the
current source. This runs on the phone through the same update flow; no computer
is needed. Verified downloads stay cached, and unchanged dependency groups are
reused. Changing only the Android SDK JAR inputs reuses AAPT2 and zipalign;
the host APK is still rebuilt. The first
upgrade from an older installation record prepares its dependencies once using
that cache. Native compilation can take more than ten minutes; the contained
platform operation allows up to one hour. A pending startup notice does not mean
that preparation has failed or that the core is already ready.

Local Pixel qualification changed the Android 36 SDK to the ext19 archive and
restored the previous SDK through ordinary Restart. The new core and SDK were
adopted, APK versionCode advanced 4 → 5 → 6 with the same signing key, and AAPT2
was reused. A repeated unchanged update reported current without changing the
host receipt. This proves local source/dependency/APK adoption; the complete
GitHub managed-update transaction and final published artifact remain under
qualification.

The installed dependency record keeps preparation incomplete until all changed
groups succeed, so an interruption cannot silently count as an adopted update.
The original Ubuntu Base image remains the installation seed. Updates can change
the package recipe and signed snapshot within Ubuntu Noble, but do not replace
the rootfs with another distribution. Restoring older source does not promise to
remove or downgrade every Linux package; actual installed package versions are
recorded separately. Preserve a full private backup for recovery that requires
restoring the entire Linux environment.

The public reference APK's publisher certificate proves its publisher identity.
Your installed APK has your installation's certificate, because it is built
locally, including any reviewed local changes. The official source provenance and
the personal APK signature are different facts. Do not install the
publisher-signed reference APK over a personally signed host and expect Android
to accept an unrelated signing key.

Ordinary compatible updates retain the application's identity and data. Deleting
the app and reinstalling it is a different operation and can erase app data and
grants. A matching certificate alone does not certify preservation of Magisk's
separate root permission. See [Android's update conditions](https://developer.android.com/google/play/app-updates).

## Backup, recovery and removal

Follow the [manual backup and recovery guide](ANDROID_RECOVERY.md) before relying
on this experimental installation. Preserve the permanent `signing/` key and
password together with `data/`, the complete Git repository, projects and the
personally signed host APK. A backup kept only on the phone does not survive
losing or wiping it.

The guide separates restoring the Linux runtime from Android app data, permissions
and a full OS backup. A local full-archive transfer and separate-directory restore
check passed; activating the restored copy remains unqualified. Uninstalling the APK alone leaves
`/data/local/ouroboros-phone`; deleting that directory also deletes the personal
runtime data and signing key still inside it.

## Reporting a problem

Include the release tag/source commit, device model, Android build, architecture,
Magisk version, failed stage and a small redacted diagnostic excerpt. State whether
the core, native host and network were individually available. Keep root setup,
APK installation, Linux provisioning and provider authorization failures separate.

Do not upload private keys, password files, API credentials, personal memory or a
populated rootfs. Other devices and general Gradle/Kotlin/NDK projects remain
unqualified; successful Java host compilation does not prove those toolchains.

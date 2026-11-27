# End-to-end smoke check for the per-user Windows Setup installer.
# Installs and uninstalls inside the current user's %LOCALAPPDATA%\Programs\Ouroboros
# and Start menu, so it refuses to run outside a Windows GitHub-hosted runner
# unless -AllowDisposableProfile is explicitly passed on a throwaway profile, and it
# needs a profile without %USERPROFILE%\Ouroboros: it seeds a synthetic checkout and
# settings there before the first install and proves every step leaves their bytes.
# Never deletes pre-existing installation state or unfamiliar files on cleanup.
param(
    [Parameter(Mandatory = $true)][string]$Installer,
    [Parameter(Mandatory = $true)][string]$Archive,
    [string]$ExpectedThumbprint = '',
    [string]$ZipSignerThumbprint = '',
    [string]$ZipSignerPublisher = '',
    [string]$SyftPath = '',
    [string]$SbomOutput = '',
    [string]$GitHubOutput = '',
    # The pinned ISCC.exe (scripts\download_inno_setup.ps1): adds step 4b, a move between two tiny
    # Setup versions compiled from ouroboros.iss. Without it that step is skipped and says so.
    [string]$Compiler = '',
    [switch]$AllowDisposableProfile
)
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false

function Test-DisposableWindowsCiRunner([bool]$OnWindows = $IsWindows) {
    $OnWindows -and ($env:GITHUB_ACTIONS -eq 'true') -and
        ($env:RUNNER_ENVIRONMENT -eq 'github-hosted') -and [bool]$env:RUNNER_TEMP
}

$IsDisposableCiRunner = Test-DisposableWindowsCiRunner
if (-not $IsDisposableCiRunner -and -not $AllowDisposableProfile) {
    throw ('smoke_windows_setup.ps1 installs into %LOCALAPPDATA%\Programs\Ouroboros; ' +
           'run only on a Windows GitHub-hosted runner or pass -AllowDisposableProfile on a throwaway user profile')
}

Add-Type -AssemblyName System.IO.Compression.FileSystem
$Repo = Split-Path -Parent $PSScriptRoot
$SetupExe = (Resolve-Path -LiteralPath $Installer).Path
$ZipArchive = (Resolve-Path -LiteralPath $Archive).Path

if ($ExpectedThumbprint) {
    $Signer = & (Join-Path $PSScriptRoot 'verify_windows_signature.ps1') -Executable $SetupExe -ExpectedThumbprint $ExpectedThumbprint
    if (($ZipSignerThumbprint -and $Signer.Thumbprint -ne $ZipSignerThumbprint) -or
        ($ZipSignerPublisher -and $Signer.Publisher -ne $ZipSignerPublisher)) {
        throw "Installer signer ($($Signer.Publisher) / $($Signer.Thumbprint)) differs from ZIP signer ($ZipSignerPublisher / $ZipSignerThumbprint)"
    }
    if ($GitHubOutput) {
        Add-Content -LiteralPath $GitHubOutput -Value "signer_thumbprint=$($Signer.Thumbprint)", "signer_publisher=$($Signer.Publisher)"
    }
}

$LocalAppData = [Environment]::GetFolderPath('LocalApplicationData')
$InstallDir = Join-Path $LocalAppData 'Programs\Ouroboros'
$ShortcutPath = Join-Path ([Environment]::GetFolderPath('Programs')) 'Ouroboros.lnk'
$UninstallReg = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\{6FB7704B-C2C5-4252-A5D0-BEC473FD007D}_is1'
$UserDataDir = Join-Path ([Environment]::GetFolderPath('UserProfile')) 'Ouroboros'

if ((Test-Path -LiteralPath $InstallDir) -or (Test-Path -LiteralPath $ShortcutPath) -or (Test-Path -LiteralPath $UninstallReg)) {
    throw 'Profile already holds an Ouroboros installation before installer smoke; refusing to modify it'
}
if (Test-Path -LiteralPath $UserDataDir) {
    throw "Profile already holds $UserDataDir; installer smoke seeds a synthetic one there and refuses to touch real data"
}

$TempRoot = if ($env:RUNNER_TEMP) { $env:RUNNER_TEMP } else { [IO.Path]::GetTempPath() }
$RunId = [guid]::NewGuid().ToString('N').Substring(0, 8)
$Work = Join-Path $TempRoot "ouroboros-setup-smoke-$RunId"
$AltDir = Join-Path $Work 'non-canonical-dir'
$ExpectedRoot = Join-Path $Work 'expected-zip'
$HomeDir = Join-Path $Work 'cli-home'
$ForeignSentinel = Join-Path $InstallDir 'portable-zip-sentinel.txt'
$KeptNote = Join-Path $InstallDir 'unfamiliar-user-note.txt'
# What a portable ZIP user already has before installing: a working checkout and settings.
$ProfileFiles = [ordered]@{
    'repo\.git\HEAD' = "ref: refs/heads/ouroboros`n"
    'repo\launcher.py' = "# synthetic checkout file $RunId`n"
    'repo\local-edit.txt' = "an uncommitted owner edit $RunId`n"
    'data\settings.json' = "{`"setup_smoke_run`": `"$RunId`"}`n"
}
$ChangingMutexName = 'Local\Ouroboros {6FB7704B-C2C5-4252-A5D0-BEC473FD007D} changing'
$CreatedForeignSentinel = $false
$CreatedKeptNote = $false
$CreatedShortcut = $false
$CreatedProfile = $false
$ProfileBefore = $null
$BusyProc = $null
$ExeLock = $null
$DatLock = $null
$ChangingMutex = $null
$LauncherHold = $null
$HeldStart = $null
$SavedUserPathCaptured = $false
$SavedUserPathExists = $false
$SavedUserPathValue = $null
$SavedUserPathKind = $null

function Get-TreeManifest([string]$Root, [string[]]$IgnoreRootFiles) {
    $Prefix = $Root.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
    $Map = [ordered]@{}
    foreach ($Item in @(Get-ChildItem -LiteralPath $Root -Recurse -Force | Sort-Object FullName)) {
        $Rel = $Item.FullName.Substring($Prefix.Length).Replace('\', '/')
        if ($IgnoreRootFiles -contains $Rel) { continue }
        $Map[$Rel] = if ($Item.PSIsContainer) { 'directory' } else {
            (Get-FileHash -LiteralPath $Item.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    }
    $Map
}

function Assert-SameTree($Expected, $Actual, [string]$What) {
    if ($Expected.Count -ne $Actual.Count) { throw "$What has $($Actual.Count) entries, expected $($Expected.Count)" }
    foreach ($Entry in $Expected.GetEnumerator()) {
        if (-not $Actual.Contains($Entry.Key) -or $Actual[$Entry.Key] -ne $Entry.Value) {
            throw "$What differs at $($Entry.Key)"
        }
    }
}

function Assert-ProfileIntact([string]$When) {
    Assert-SameTree $ProfileBefore (Get-TreeManifest $UserDataDir @()) "The pre-existing profile $When"
}

function Assert-Shortcut([string]$Path, [string]$ExpectedTarget) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "Start menu shortcut missing: $Path" }
    $Wsh = New-Object -ComObject WScript.Shell
    $Link = $Wsh.CreateShortcut($Path)
    if (-not [String]::Equals($Link.TargetPath, $ExpectedTarget, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Shortcut target '$($Link.TargetPath)' is not '$ExpectedTarget'"
    }
    $Shell = New-Object -ComObject Shell.Application
    $Folder = $Shell.Namespace((Split-Path -Parent $Path))
    $Item = if ($Folder) { $Folder.ParseName((Split-Path -Leaf $Path)) } else { $null }
    if (-not $Item) { throw "Could not inspect Start menu shortcut via Shell.Application: $Path" }
    $Aumid = [string]$Item.ExtendedProperty('System.AppUserModel.ID')
    if ($Aumid -ne 'com.ouroboros.agent') {
        throw "Shortcut AppUserModelID is '$Aumid', not 'com.ouroboros.agent'"
    }
}

function Invoke-SilentSetup([string[]]$ExtraArgs, [string]$LogFile) {
    $SetupArgs = @('/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART', '/SP-', "/LOG=`"$LogFile`"") + $ExtraArgs
    Start-Process -FilePath $SetupExe -Wait -PassThru -ArgumentList $SetupArgs
}

function Start-ExactProcess([string]$FilePath, [string[]]$ArgumentList) {
    $Psi = [Diagnostics.ProcessStartInfo]::new()
    $Psi.FileName = $FilePath
    $Psi.UseShellExecute = $false
    foreach ($Arg in $ArgumentList) { [void]$Psi.ArgumentList.Add($Arg) }
    $Proc = [Diagnostics.Process]::Start($Psi)
    if (-not $Proc) { throw "Failed to start process: $FilePath" }
    $Proc
}

function Invoke-CmdInDir([string]$WorkingDir, [string]$CmdPath, [string[]]$CmdArgs) {
    if (-not (Test-Path -LiteralPath $CmdPath -PathType Leaf)) { throw "CLI script missing: $CmdPath" }
    Push-Location -LiteralPath $WorkingDir
    try {
        $ErrorActionPreference = 'Continue'
        $global:LASTEXITCODE = 0
        $Out = & $CmdPath @CmdArgs 2>&1
        [pscustomobject]@{ ExitCode = $LASTEXITCODE; Output = ($Out | Out-String) }
    } finally {
        Pop-Location
    }
}

function Open-FileWithRetry {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][IO.FileAccess]$Access,
        [Parameter(Mandatory = $true)][IO.FileShare]$Share,
        [Parameter(Mandatory = $true)][string]$Phase,
        [int]$TimeoutMs = 15000,
        [int]$PollMs = 100
    )
    $Sw = [Diagnostics.Stopwatch]::StartNew()
    $Attempts = 0
    $Leaf = Split-Path -Leaf $Path
    while ($true) {
        $Attempts += 1
        try {
            $Stream = [IO.File]::Open($Path, [IO.FileMode]::Open, $Access, $Share)
            Write-Host ("{0}: opened {1} in {2} ms (attempts={3})" -f $Phase, $Leaf, $Sw.ElapsedMilliseconds, $Attempts)
            return $Stream
        } catch {
            $Err = $_.Exception
            while ($Err -is [System.Management.Automation.MethodInvocationException] -and $Err.InnerException) {
                $Err = $Err.InnerException
            }
            $Hr = $Err.HResult
            $IsSharingLock = ($Err -is [IO.IOException]) -and ($Hr -eq -2147024864 -or $Hr -eq -2147024863)
            $ElapsedMs = $Sw.ElapsedMilliseconds
            if (-not $IsSharingLock) {
                throw ("{0}: {1} failed to open after {2} ms (attempts={3}, type={4}, hresult=0x{5:X8})" -f `
                    $Phase, $Leaf, $ElapsedMs, $Attempts, $Err.GetType().FullName, ($Hr -band 0xFFFFFFFFL))
            }
            $RemainingMs = [int64]$TimeoutMs - $ElapsedMs
            if ($RemainingMs -le 0) {
                throw ("{0}: {1} remained locked after {2} ms (attempts={3}, type={4}, hresult=0x{5:X8})" -f `
                    $Phase, $Leaf, $ElapsedMs, $Attempts, $Err.GetType().FullName, ($Hr -band 0xFFFFFFFFL))
            }
            $SleepMs = [int][Math]::Min([int64]$PollMs, $RemainingMs)
            if ($SleepMs -gt 0) { Start-Sleep -Milliseconds $SleepMs }
        }
    }
}

function Assert-ExclusiveOpen([string]$Path) {
    [IO.File]::Open($Path, [IO.FileMode]::Open, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None).Dispose()
}

function Assert-UninstallRefusesPromptly([string]$UninstPath, [string]$Why) {
    # Omit /SUPPRESSMSGBOXES under a bounded wait so any retry dialog holding unins*.dat fails.
    $Proc = Start-ExactProcess $UninstPath @('/VERYSILENT', '/NORESTART')
    try {
        if (-not $Proc.WaitForExit(30000)) {
            try { $Proc.Kill($true); $Proc.WaitForExit(10000) | Out-Null } catch { }
            throw "Uninstall did not exit promptly $Why"
        }
        if ($Proc.ExitCode -eq 0) { throw "Uninstall unexpectedly succeeded $Why" }
    } finally {
        try { $Proc.Dispose() } catch { }
    }
}

function Remove-DirOnlyIfEmpty([string]$Dir) {
    if ((Test-Path -LiteralPath $Dir -PathType Container) -and @(Get-ChildItem -LiteralPath $Dir -Force).Count -eq 0) {
        Remove-Item -LiteralPath $Dir -Force
    }
}

try {
    New-Item -ItemType Directory -Force -Path $Work | Out-Null

    # 1. Refuse non-canonical /DIR without installing or writing state.
    $AltProc = Invoke-SilentSetup @("/DIR=`"$AltDir`"") (Join-Path $Work 'non-canonical.log')
    if ($AltProc.ExitCode -eq 0) { throw 'Setup unexpectedly accepted a non-canonical /DIR' }
    if ((Test-Path -LiteralPath (Join-Path $AltDir 'Ouroboros.exe')) -or
        (Test-Path -LiteralPath $ShortcutPath) -or (Test-Path -LiteralPath $UninstallReg)) {
        throw 'Setup modified state when refusing a non-canonical /DIR'
    }

    # 2. Refuse a foreign non-empty folder (such as an unpacked portable ZIP) without modifying it.
    New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
    Set-Content -LiteralPath $ForeignSentinel -Value 'keep-foreign-folder' -NoNewline
    $CreatedForeignSentinel = $true
    $ForeignProc = Invoke-SilentSetup @() (Join-Path $Work 'foreign.log')
    if ($ForeignProc.ExitCode -eq 0) { throw 'Setup unexpectedly installed over a foreign non-empty folder' }
    if ((Get-Content -Raw -LiteralPath $ForeignSentinel) -ne 'keep-foreign-folder' -or
        (Test-Path -LiteralPath (Join-Path $InstallDir 'Ouroboros.exe')) -or
        (Test-Path -LiteralPath $ShortcutPath) -or (Test-Path -LiteralPath $UninstallReg)) {
        throw 'Setup modified state when refusing a foreign folder'
    }
    Remove-Item -LiteralPath $ForeignSentinel -Force
    $CreatedForeignSentinel = $false
    if (@(Get-ChildItem -LiteralPath $InstallDir -Force).Count -ne 0) {
        throw "Refusing to wipe unexpected entries left in $InstallDir after foreign-folder test"
    }
    Remove-DirOnlyIfEmpty $InstallDir

    # 3. Refuse a first install over the owner's own Start shortcut (one made for a portable ZIP) without touching it.
    $PortableExe = Join-Path $Work 'portable\Ouroboros\Ouroboros.exe'
    New-Item -ItemType File -Force -Path $PortableExe | Out-Null
    $Wsh = New-Object -ComObject WScript.Shell
    $ForeignLink = $Wsh.CreateShortcut($ShortcutPath)
    $ForeignLink.TargetPath = $PortableExe
    $ForeignLink.Arguments = "--portable-shortcut-$RunId"
    $ForeignLink.Save()
    $CreatedShortcut = $true
    $ForeignLinkHash = (Get-FileHash -LiteralPath $ShortcutPath -Algorithm SHA256).Hash
    $ShortcutProc = Invoke-SilentSetup @() (Join-Path $Work 'foreign-shortcut.log')
    if ($ShortcutProc.ExitCode -eq 0) { throw "Setup unexpectedly installed over the owner's Start shortcut" }
    if ((Get-FileHash -LiteralPath $ShortcutPath -Algorithm SHA256).Hash -ne $ForeignLinkHash -or
        (Test-Path -LiteralPath (Join-Path $InstallDir 'Ouroboros.exe')) -or (Test-Path -LiteralPath $UninstallReg)) {
        throw "Setup modified state when refusing the owner's Start shortcut"
    }
    Remove-Item -LiteralPath $ShortcutPath -Force
    $CreatedShortcut = $false
    Remove-DirOnlyIfEmpty $InstallDir

    # 4. A portable user's checkout and settings exist before the first install and keep their bytes through it.
    foreach ($Entry in $ProfileFiles.GetEnumerator()) {
        $Path = Join-Path $UserDataDir $Entry.Key
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Path) | Out-Null
        $CreatedProfile = $true
        [IO.File]::WriteAllText($Path, $Entry.Value)
    }
    $ProfileBefore = Get-TreeManifest $UserDataDir @()

    # 4b. A normal update between two distinct payloads, each delivered as its own Setup: the new
    #     version's files land byte for byte, a folder only the old version shipped (as a Chromium
    #     build in its version-named folder) stays until uninstall, which then removes both
    #     versions' files, and unfamiliar files and owner data stay throughout. Uninstalling a
    #     copy without unfamiliar files removes the install folder itself.
    if ($Compiler) {
        $Transition = Join-Path $Work 'transition'
        $Payloads = [ordered]@{
            '7.98.0' = [ordered]@{ '_internal\changed.txt' = 'payload 7.98.0'; '_internal\retired-in-7.98.1\browser.txt' = 'old build' }
            '7.98.1' = [ordered]@{ '_internal\changed.txt' = 'payload 7.98.1'; '_internal\added-in-7.98.1.txt' = 'new file' }
        }
        $Setups = @{}
        foreach ($Version in $Payloads.Keys) {
            $Root = Join-Path $Transition "payload-$Version\Ouroboros"
            $Files = [ordered]@{ 'Ouroboros.exe' = "transition fixture $Version"; '_internal\VERSION' = $Version
                                 '_internal\python-standalone\python.exe' = "fixture interpreter $Version"
                                 'bin\ouroboros.cmd' = "@rem $Version" }
            foreach ($Entry in $Payloads[$Version].GetEnumerator()) { $Files[$Entry.Key] = $Entry.Value }
            foreach ($Entry in $Files.GetEnumerator()) {
                $Path = Join-Path $Root $Entry.Key
                New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Path) | Out-Null
                [IO.File]::WriteAllText($Path, $Entry.Value)
            }
            $Out = Join-Path $Transition "setup-$Version"
            & (Join-Path $PSScriptRoot 'pack_windows_setup.ps1') -PayloadRoot $Root -OutputDir $Out -Compiler $Compiler
            $Setups[$Version] = @{ Exe = (Join-Path $Out "Ouroboros-Setup-$Version-windows-x64.exe"); Root = $Root }
        }
        $TransitionNote = Join-Path $InstallDir 'transition-user-note.txt'
        $V1Proc = Start-Process -FilePath $Setups['7.98.0'].Exe -Wait -PassThru -ArgumentList @(
            '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART', '/SP-', "/LOG=`"$(Join-Path $Transition 'install-7.98.0.log')`"")
        if ($V1Proc.ExitCode -ne 0) { throw "Transition fixture Setup 7.98.0 exited $($V1Proc.ExitCode)" }
        if ((Get-ItemProperty -LiteralPath $UninstallReg).DisplayVersion -ne '7.98.0') {
            throw 'After Setup 7.98.0 the registration names another version'
        }
        Set-Content -LiteralPath $TransitionNote -Value "keep $RunId" -NoNewline
        $NoteHash = (Get-FileHash -LiteralPath $TransitionNote -Algorithm SHA256).Hash
        # Losing operations while another changer holds the mutex leave v1 intact and unins*.dat unlocked.
        $V1Tree = Get-TreeManifest $InstallDir @('unins000.exe', 'unins000.dat', 'transition-user-note.txt')
        $V1Uninst = (Get-ItemProperty -LiteralPath $UninstallReg).UninstallString.Trim().Trim('"')
        $ChangingMutex = [Threading.Mutex]::new($false, $ChangingMutexName)
        try {
            $LosingV2 = Start-Process -FilePath $Setups['7.98.1'].Exe -Wait -PassThru -ArgumentList @(
                '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART', '/SP-', "/LOG=`"$(Join-Path $Transition 'losing-7.98.1.log')`"")
            if ($LosingV2.ExitCode -eq 0) { throw 'Losing Setup 7.98.1 unexpectedly succeeded while change mutex was held' }
            Assert-UninstallRefusesPromptly $V1Uninst 'while change mutex was held on v1'
        } finally {
            $ChangingMutex.Dispose()
            $ChangingMutex = $null
        }
        if ((Get-ItemProperty -LiteralPath $UninstallReg).DisplayVersion -ne '7.98.0') {
            throw 'Losing operations altered v1 registration'
        }
        Assert-SameTree $V1Tree (Get-TreeManifest $InstallDir @('unins000.exe', 'unins000.dat', 'transition-user-note.txt')) 'v1 after losing operations'
        Assert-ExclusiveOpen ([IO.Path]::ChangeExtension($V1Uninst, '.dat'))
        Assert-ProfileIntact 'after losing operations on v1'
        # Subsequent winning v2 install and uninstall complete cleanly.
        $V2Proc = Start-Process -FilePath $Setups['7.98.1'].Exe -Wait -PassThru -ArgumentList @(
            '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART', '/SP-', "/LOG=`"$(Join-Path $Transition 'install-7.98.1.log')`"")
        if ($V2Proc.ExitCode -ne 0) { throw "Transition fixture Setup 7.98.1 exited $($V2Proc.ExitCode)" }
        if ((Get-ItemProperty -LiteralPath $UninstallReg).DisplayVersion -ne '7.98.1') {
            throw 'After Setup 7.98.1 the registration names another version'
        }
        if (-not (Test-Path -LiteralPath $TransitionNote -PathType Leaf) -or
            (Get-FileHash -LiteralPath $TransitionNote -Algorithm SHA256).Hash -ne $NoteHash) {
            throw 'The update changed or removed an unfamiliar file'
        }
        Assert-SameTree (Get-TreeManifest $Setups['7.98.1'].Root @()) (Get-TreeManifest $InstallDir @(
            'unins000.exe', 'unins000.dat', 'transition-user-note.txt', '_internal/retired-in-7.98.1',
            '_internal/retired-in-7.98.1/browser.txt')) 'The updated payload'
        if ((Get-Content -Raw -LiteralPath (Join-Path $InstallDir '_internal\retired-in-7.98.1\browser.txt')) -ne 'old build') {
            throw 'The update did not keep the folder only the old version shipped until uninstall'
        }
        Assert-Shortcut $ShortcutPath (Join-Path $InstallDir 'Ouroboros.exe')
        Assert-ProfileIntact 'after the version update'
        $TransitionUninst = (Get-ItemProperty -LiteralPath $UninstallReg).UninstallString.Trim().Trim('"')
        $Proc = Start-Process -FilePath $TransitionUninst -Wait -PassThru -ArgumentList @('/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART')
        if ($Proc.ExitCode -ne 0) { throw "Transition fixture uninstall exited $($Proc.ExitCode)" }
        for ($Tick = 0; $Tick -lt 150 -and (Test-Path -LiteralPath $TransitionUninst); $Tick++) { Start-Sleep -Milliseconds 200 }
        if (((@(Get-ChildItem -LiteralPath $InstallDir -Force | ForEach-Object { $_.Name })) -join ',') -ne 'transition-user-note.txt' -or
            (Test-Path -LiteralPath $ShortcutPath) -or (Test-Path -LiteralPath $UninstallReg)) {
            throw 'Uninstalling after the update left files of either version, the shortcut or the registration'
        }
        Remove-Item -LiteralPath $TransitionNote -Force
        Remove-DirOnlyIfEmpty $InstallDir
        Assert-ProfileIntact 'after the version update was uninstalled'
        $Proc = Start-Process -FilePath $Setups['7.98.1'].Exe -Wait -PassThru -ArgumentList @(
            '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART', '/SP-', "/LOG=`"$(Join-Path $Transition 'install-clean.log')`"")
        if ($Proc.ExitCode -ne 0) { throw "Clean fixture Setup exited $($Proc.ExitCode)" }
        $CleanUninst = (Get-ItemProperty -LiteralPath $UninstallReg).UninstallString.Trim().Trim('"')
        $CleanLog = Join-Path $Transition 'uninstall-clean.log'
        $Proc = Start-Process -FilePath $CleanUninst -Wait -PassThru -ArgumentList @(
            '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART', "/LOG=`"$CleanLog`"")
        if ($Proc.ExitCode -ne 0) { throw "Clean fixture uninstall exited $($Proc.ExitCode)" }
        for ($Tick = 0; $Tick -lt 150 -and (Test-Path -LiteralPath $InstallDir); $Tick++) { Start-Sleep -Milliseconds 200 }
        if ((Test-Path -LiteralPath $InstallDir) -or (Test-Path -LiteralPath $ShortcutPath) -or (Test-Path -LiteralPath $UninstallReg)) {
            throw 'Uninstalling a copy without unfamiliar files left its install folder, the shortcut or the registration'
        }
        Assert-ProfileIntact 'after a clean copy was uninstalled'
    } else {
        Write-Host 'Step 4b skipped: pass -Compiler <pinned ISCC.exe> to test an update between two Setup versions.'
    }

    # 5. Fresh silent per-user install into %LOCALAPPDATA%\Programs\Ouroboros.
    $InstallLog = Join-Path $Work 'install.log'
    $InstallProc = Invoke-SilentSetup @() $InstallLog
    if ($InstallProc.ExitCode -ne 0) {
        if (Test-Path -LiteralPath $InstallLog) { Get-Content -LiteralPath $InstallLog | Write-Host }
        throw "Silent Windows install exited $($InstallProc.ExitCode)"
    }
    Assert-ProfileIntact 'after the first install'
    # unins*.exe/.dat prove Setup's ownership; the plain-source leaf is the entry its handoffs run any checkout through.
    foreach ($Meta in @('unins000.exe', 'unins000.dat', '_internal\ouroboros\launcher_windows_runtime.py')) {
        if (-not (Test-Path -LiteralPath (Join-Path $InstallDir $Meta) -PathType Leaf)) {
            throw "Installed folder is missing $Meta"
        }
    }

    # 6. Full installed tree must match the portable ZIP payload byte for byte.
    [IO.Compression.ZipFile]::ExtractToDirectory($ZipArchive, $ExpectedRoot)
    $ExpectedTree = Get-TreeManifest (Join-Path $ExpectedRoot 'Ouroboros') @()
    Assert-SameTree $ExpectedTree (Get-TreeManifest $InstallDir @('unins000.exe', 'unins000.dat')) 'The installed payload'
    if ($ExpectedThumbprint) {
        $InstalledSigners = @(& (Join-Path $PSScriptRoot 'verify_windows_signature.ps1') `
            -ExtractedRoot (Split-Path -Parent $InstallDir) -ExpectedThumbprint $ExpectedThumbprint)
        if ($InstalledSigners.Count -ne 3) {
            throw "Expected 3 signed installed payload executables, verified $($InstalledSigners.Count)"
        }
    }

    # 7. Start menu shortcut target, System.AppUserModel.ID and HKCU Uninstall entry.
    $ExpectedExe = Join-Path $InstallDir 'Ouroboros.exe'
    $ExpectedExeHash = (Get-FileHash -LiteralPath $ExpectedExe -Algorithm SHA256).Hash.ToLowerInvariant()
    Assert-Shortcut $ShortcutPath $ExpectedExe
    if (-not (Test-Path -LiteralPath $UninstallReg)) { throw "HKCU uninstall entry missing: $UninstallReg" }
    $Reg = Get-ItemProperty -LiteralPath $UninstallReg
    $InstalledVersion = (Get-Content -Raw -LiteralPath (Join-Path $InstallDir '_internal\VERSION')).Trim()
    if ($Reg.DisplayName -ne 'Ouroboros' -or $Reg.DisplayVersion -ne $InstalledVersion -or
        $Reg.UninstallString -notlike '*unins000.exe*') {
        throw "Unexpected HKCU uninstall metadata: $($Reg.DisplayName) / $($Reg.DisplayVersion) / $($Reg.UninstallString)"
    }
    $UninstExe = $Reg.UninstallString.Trim().Trim('"')

    # 8. Refused upgrade & uninstall while target binaries are busy (GUI-handoff pythonw.exe
    #    running + Ouroboros.exe locked) must exit non-zero, never close the process, and
    #    preserve the previous EXE, shortcut, registration, unfamiliar file and owner data.
    Set-Content -LiteralPath $KeptNote -Value 'keep-unfamiliar-file' -NoNewline
    $CreatedKeptNote = $true

    $PythonwExe = Join-Path $InstallDir '_internal\python-standalone\pythonw.exe'
    $BusyReady = Join-Path $Work 'pythonw-busy.ready'
    $BusyProc = Start-ExactProcess $PythonwExe @(
        '-I', '-B', '-c',
        "import pathlib,sys,time; pathlib.Path(sys.argv[1]).write_text('up'); time.sleep(120)",
        $BusyReady)
    for ($Tick = 0; $Tick -lt 150; $Tick++) {
        if (Test-Path -LiteralPath $BusyReady -PathType Leaf) { break }
        if ($BusyProc.HasExited) { throw "Busy-probe pythonw.exe exited early ($($BusyProc.ExitCode))" }
        Start-Sleep -Milliseconds 100
    }
    if (-not (Test-Path -LiteralPath $BusyReady -PathType Leaf)) { throw 'Busy-probe pythonw.exe did not become ready' }

    $BusyUpgrade = Invoke-SilentSetup @() (Join-Path $Work 'busy-upgrade.log')
    if ($BusyUpgrade.ExitCode -eq 0) { throw 'Setup unexpectedly upgraded while installed pythonw.exe was running' }
    if ($BusyProc.HasExited) { throw 'Setup unexpectedly closed the running pythonw.exe process' }

    $BusyUninstall = Start-Process -FilePath $UninstExe -Wait -PassThru -ArgumentList @('/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART')
    if ($BusyUninstall.ExitCode -eq 0) { throw 'Uninstall unexpectedly succeeded while installed pythonw.exe was running' }
    if ($BusyProc.HasExited) { throw 'Uninstall unexpectedly closed the running pythonw.exe process' }

    $BusyProc.Kill()
    $BusyProc.WaitForExit(10000) | Out-Null
    $BusyProc.Dispose()
    $BusyProc = $null

    # Also verify a direct OS file lock on Ouroboros.exe or unins*.dat refuses upgrade silently.
    $ExeLock = Open-FileWithRetry -Path $ExpectedExe -Access ([IO.FileAccess]::Read) -Share ([IO.FileShare]::Read) `
        -Phase 'step 8 locked-exe-upgrade setup'
    try {
        $LockedExeUpgrade = Invoke-SilentSetup @() (Join-Path $Work 'locked-exe-upgrade.log')
        if ($LockedExeUpgrade.ExitCode -eq 0) { throw 'Setup unexpectedly upgraded while Ouroboros.exe was locked' }
    } finally {
        $ExeLock.Dispose()
        $ExeLock = $null
    }

    $UninstDat = [IO.Path]::ChangeExtension($UninstExe, '.dat')
    $DatLock = [IO.File]::Open($UninstDat, [IO.FileMode]::Open, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
    try {
        $LockedDatUpgrade = Invoke-SilentSetup @() (Join-Path $Work 'locked-dat-upgrade.log')
        if ($LockedDatUpgrade.ExitCode -eq 0) { throw 'Setup unexpectedly upgraded while unins000.dat was locked' }
    } finally {
        $DatLock.Dispose()
        $DatLock = $null
    }
    if (Test-Path -LiteralPath (Join-Path $InstallDir 'unins001.dat')) {
        throw 'Refused upgrade created a duplicate unins001.dat while unins000.dat was locked'
    }

    # Mutual exclusion: while another changer holds SetupChangingName, both Setup and Uninstall refuse.
    $ChangingMutex = [Threading.Mutex]::new($false, $ChangingMutexName)
    try {
        $MutexUpgrade = Invoke-SilentSetup @() (Join-Path $Work 'mutex-upgrade.log')
        if ($MutexUpgrade.ExitCode -eq 0) { throw 'Setup unexpectedly proceeded while the change mutex was already held' }
        Assert-UninstallRefusesPromptly $UninstExe 'while the change mutex was held'
        Assert-ExclusiveOpen $UninstDat
    } finally {
        $ChangingMutex.Dispose()
        $ChangingMutex = $null
    }

    if (((Get-FileHash -LiteralPath $ExpectedExe -Algorithm SHA256).Hash.ToLowerInvariant() -ne $ExpectedExeHash) -or
        -not (Test-Path -LiteralPath $UninstExe -PathType Leaf) -or
        -not (Test-Path -LiteralPath ([IO.Path]::ChangeExtension($UninstExe, '.dat')) -PathType Leaf) -or
        -not (Test-Path -LiteralPath $UninstallReg) -or
        (Get-Content -Raw -LiteralPath $KeptNote) -ne 'keep-unfamiliar-file') {
        throw 'Refused upgrade/uninstall did not preserve the existing installation intact'
    }
    Assert-ProfileIntact 'after the refused upgrade and uninstall'
    Assert-Shortcut $ShortcutPath $ExpectedExe

    # 9. Once nothing is in use, Setup and Uninstall hold Ouroboros.exe and the CLI's python.exe
    #    (ouroboros.iss HoldStartPrograms; step 4b checks how long): read access shared only for
    #    writing and deletion. The same hold on a copy of a system program makes Windows refuse to
    #    start it, as it refuses a Start click or a CLI command on Ouroboros then, while deleting it
    #    still works.
    $HeldExe = Join-Path $Work 'launcher-hold\held.exe'
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $HeldExe) | Out-Null
    Copy-Item -LiteralPath (Join-Path ([Environment]::SystemDirectory) 'whoami.exe') -Destination $HeldExe
    $LauncherHold = [IO.File]::Open($HeldExe, [IO.FileMode]::Open, [IO.FileAccess]::Read,
        ([IO.FileShare]::Write -bor [IO.FileShare]::Delete))
    $HeldRefusal = $null
    try { $HeldStart = Start-ExactProcess $HeldExe @() } catch { $HeldRefusal = $_.Exception }
    if ($HeldStart) { throw 'Windows started a program held the way Uninstall holds Ouroboros.exe' }
    while ($HeldRefusal -and $HeldRefusal -isnot [ComponentModel.Win32Exception]) { $HeldRefusal = $HeldRefusal.InnerException }
    if (-not $HeldRefusal -or $HeldRefusal.NativeErrorCode -ne 32) {
        throw "Starting the held program did not fail with a sharing violation: $HeldRefusal"
    }
    [IO.File]::Delete($HeldExe)
    $LauncherHold.Dispose()
    $LauncherHold = $null
    if (Test-Path -LiteralPath $HeldExe) { throw 'The held program remained after its deletion and the end of its hold' }

    # 10. Repair: reinstalling the same Setup over an installed tree that no longer matches it (one
    #     installer-owned file changed, one missing) lands its payload exactly and keeps unfamiliar
    #     files and owner data. A move between two versions is step 4b (-Compiler).
    $ChangedFile = Join-Path $InstallDir 'bin\ouroboros.cmd'
    $MissingFile = Join-Path $InstallDir '_internal\VERSION'
    [IO.File]::AppendAllText($ChangedFile, "rem changed by an older payload $RunId`r`n")
    Remove-Item -LiteralPath $MissingFile -Force
    $UpgradeLog = Join-Path $Work 'upgrade.log'
    $UpgradeProc = Invoke-SilentSetup @() $UpgradeLog
    if ($UpgradeProc.ExitCode -ne 0) {
        if (Test-Path -LiteralPath $UpgradeLog) { Get-Content -LiteralPath $UpgradeLog | Write-Host }
        throw "In-place Windows upgrade exited $($UpgradeProc.ExitCode)"
    }
    Assert-SameTree $ExpectedTree (Get-TreeManifest $InstallDir @('unins000.exe', 'unins000.dat', 'unfamiliar-user-note.txt')) `
        'The reinstalled payload'
    if ((Get-Content -Raw -LiteralPath $KeptNote) -ne 'keep-unfamiliar-file') {
        throw 'In-place upgrade altered an unfamiliar file'
    }
    Assert-ProfileIntact 'after the reinstall'
    Assert-Shortcut $ShortcutPath $ExpectedExe
    $UninstExe = (Get-ItemProperty -LiteralPath $UninstallReg).UninstallString.Trim().Trim('"')

    # 11. Installed CLI (from ordinary and old-layout working folders), documented portable -> installed
    #     CLI shim migration (--force) + actual shim call, non-blocking GUI refusal, Betterleaks smoke,
    #     and optional installed-tree CycloneDX SBOM.
    $OrdinaryCwd = Join-Path $Work 'ordinary-cwd'
    $OldCwd = Join-Path $Work 'old-layout-cwd'
    $CliShimTarget = Join-Path $Work 'cli-shim-target'
    $MigratedShim = Join-Path $CliShimTarget 'ouroboros.cmd'
    $CliAppRoot = Join-Path $HomeDir 'Ouroboros'
    New-Item -ItemType Directory -Force -Path $HomeDir, $OrdinaryCwd, (Join-Path $OldCwd 'ouroboros') | Out-Null
    # Legitimate old-layout working folder (empty ouroboros package without packaged_cli; no code to run from cwd).
    [IO.File]::WriteAllText((Join-Path $OldCwd 'ouroboros\__init__.py'), '')
    $InstalledCli = Join-Path $InstallDir 'bin\ouroboros.cmd'
    $InstalledCliInstaller = Join-Path $InstallDir 'bin\install-ouroboros-cli.cmd'
    $PortableCliInstaller = Join-Path $ExpectedRoot 'Ouroboros\bin\install-ouroboros-cli.cmd'
    $SavedEnv = @{}
    foreach ($Key in @('OUROBOROS_APP_ROOT', 'OUROBOROS_REPO_DIR', 'OUROBOROS_DATA_DIR',
                       'OUROBOROS_SETTINGS_PATH', 'OUROBOROS_PID_FILE', 'OUROBOROS_PORT_FILE')) {
        $SavedEnv[$Key] = [Environment]::GetEnvironmentVariable($Key, 'Process')
    }
    try {
        foreach ($Key in $SavedEnv.Keys) { [Environment]::SetEnvironmentVariable($Key, $null, 'Process') }
        $env:OUROBOROS_APP_ROOT = $CliAppRoot
        # Snapshot HKCU:\Environment\Path (unexpanded) before install-ouroboros-cli.cmd so finally restores it cleanly.
        $EnvKey = [Microsoft.Win32.Registry]::CurrentUser.OpenSubKey('Environment', $false)
        try {
            if ($EnvKey -and ($EnvKey.GetValueNames() -contains 'Path')) {
                $SavedUserPathExists = $true
                $SavedUserPathKind = $EnvKey.GetValueKind('Path')
                $SavedUserPathValue = $EnvKey.GetValue('Path', '', [Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames)
            }
            $SavedUserPathCaptured = $true
        } finally {
            if ($EnvKey) { $EnvKey.Dispose() }
        }
        # Seed an owned CLI shim from the portable copy in a temp target dir (does not touch $CliAppRoot).
        $PortableShimInstall = Invoke-CmdInDir $OrdinaryCwd $PortableCliInstaller @('--target-dir', $CliShimTarget)
        if ($PortableShimInstall.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $MigratedShim -PathType Leaf)) {
            throw "Portable CLI shim install failed: exit $($PortableShimInstall.ExitCode)`n$($PortableShimInstall.Output)"
        }
        # While Setup or Uninstall holds its mutex, the installed CLI and Ouroboros.exe (ordinary Start and automatic
        # start alike, with no modal dialog) exit 75 before any work, while the portable shim still runs.
        $ChangingMutex = [Threading.Mutex]::new($false, $ChangingMutexName)
        try {
            foreach ($Dir in @($OrdinaryCwd, $OldCwd)) {
                $CliRefusal = Invoke-CmdInDir $Dir $InstalledCli @('--help')
                if ($CliRefusal.ExitCode -ne 75 -or $CliRefusal.Output -notmatch 'changing this installation') {
                    throw "Installed CLI from $Dir did not refuse while Setup held the mutex: exit $($CliRefusal.ExitCode)"
                }
            }
            foreach ($GuiCase in @(
                @{ Label = 'ordinary Start'; Args = [string[]]@() }
                @{ Label = 'automatic start'; Args = [string[]]@('--launch-intent', 'automatic') }
            )) {
                $GuiProc = Start-ExactProcess $ExpectedExe $GuiCase.Args
                try {
                    if (-not $GuiProc.WaitForExit(30000)) {
                        try { $GuiProc.Kill($true); $GuiProc.WaitForExit(10000) | Out-Null } catch { }
                        throw "Ouroboros.exe ($($GuiCase.Label)) did not exit promptly while Setup held the mutex"
                    }
                    if ($GuiProc.ExitCode -ne 75 -or (Test-Path -LiteralPath $CliAppRoot)) {
                        throw "Ouroboros.exe ($($GuiCase.Label)) did not refuse before work: exit $($GuiProc.ExitCode)"
                    }
                } finally {
                    try { $GuiProc.Dispose() } catch { }
                }
                # Neither launch mode may keep Ouroboros.exe mapped waiting for a dialog.
                $GuiReleaseStream = Open-FileWithRetry -Path $ExpectedExe -Access ([IO.FileAccess]::ReadWrite) `
                    -Share ([IO.FileShare]::None) -Phase "step 11 post-gui-exit ($($GuiCase.Label))"
                $GuiReleaseStream.Dispose()
            }
            $PortableShimRun = Invoke-CmdInDir $OrdinaryCwd $MigratedShim @('--help')
            if ($PortableShimRun.ExitCode -ne 0) {
                throw "Portable CLI shim refused while Setup held the mutex: exit $($PortableShimRun.ExitCode)`n$($PortableShimRun.Output)"
            }
        } finally {
            $ChangingMutex.Dispose()
            $ChangingMutex = $null
        }
        foreach ($Dir in @($OrdinaryCwd, $OldCwd)) {
            $CliRun = Invoke-CmdInDir $Dir $InstalledCli @('--help')
            if ($CliRun.ExitCode -ne 0) { throw "Installed CLI smoke from $Dir failed: $($CliRun.ExitCode)`n$($CliRun.Output)" }
        }
        # Documented migration from portable -> installed requires --force while the portable copy still exists,
        # then calling the migrated shim actually runs the installed command (exits 0 normally, 75 under mutex).
        $NoForce = Invoke-CmdInDir $OrdinaryCwd $InstalledCliInstaller @('--target-dir', $CliShimTarget)
        if ($NoForce.ExitCode -eq 0) { throw 'Installed CLI installer overwrote a live portable shim without --force' }
        $MigratedInstall = Invoke-CmdInDir $OldCwd $InstalledCliInstaller @('--force', '--target-dir', $CliShimTarget)
        if ($MigratedInstall.ExitCode -ne 0) {
            throw "CLI shim migration (--force) from $OldCwd failed: exit $($MigratedInstall.ExitCode)`n$($MigratedInstall.Output)"
        }
        foreach ($Dir in @($OrdinaryCwd, $OldCwd)) {
            $ShimRun = Invoke-CmdInDir $Dir $MigratedShim @('--help')
            if ($ShimRun.ExitCode -ne 0) { throw "Migrated CLI shim from $Dir failed: exit $($ShimRun.ExitCode)`n$($ShimRun.Output)" }
        }
        $ChangingMutex = [Threading.Mutex]::new($false, $ChangingMutexName)
        try {
            foreach ($Dir in @($OrdinaryCwd, $OldCwd)) {
                $MigratedRefusal = Invoke-CmdInDir $Dir $MigratedShim @('--help')
                if ($MigratedRefusal.ExitCode -ne 75 -or $MigratedRefusal.Output -notmatch 'changing this installation') {
                    throw "Migrated CLI shim from $Dir did not execute the installed copy under mutex: exit $($MigratedRefusal.ExitCode)"
                }
            }
        } finally {
            $ChangingMutex.Dispose()
            $ChangingMutex = $null
        }
    } finally {
        foreach ($Key in $SavedEnv.Keys) { [Environment]::SetEnvironmentVariable($Key, $SavedEnv[$Key], 'Process') }
    }
    python (Join-Path $Repo 'scripts\betterleaks_platform_smoke.py') --bundle-root (Join-Path $InstallDir '_internal')
    if ($LASTEXITCODE -ne 0) { throw "Installed Betterleaks runtime smoke failed: $LASTEXITCODE" }
    if ($SyftPath -and $SbomOutput) {
        & $SyftPath "dir:$InstallDir" --output "cyclonedx-json=$SbomOutput"
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $SbomOutput -PathType Leaf)) {
            throw "Failed to generate CycloneDX SBOM for installed Windows tree: $LASTEXITCODE"
        }
    }

    # 12. Ordinary silent uninstall removes installed files, shortcut and registry
    #     while preserving unfamiliar files and %USERPROFILE%\Ouroboros.
    $UninstProc = Start-Process -FilePath $UninstExe -Wait -PassThru -ArgumentList @('/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART')
    if ($UninstProc.ExitCode -ne 0) { throw "Silent Windows uninstall exited $($UninstProc.ExitCode)" }
    for ($Tick = 0; $Tick -lt 150; $Tick++) {
        if (-not (Test-Path -LiteralPath $UninstExe)) { break }
        Start-Sleep -Milliseconds 200
    }
    foreach ($Gone in @($ExpectedExe, (Join-Path $InstallDir '_internal'), (Join-Path $InstallDir 'bin'),
                        $UninstExe, ([IO.Path]::ChangeExtension($UninstExe, '.dat')), $ShortcutPath, $UninstallReg)) {
        if (Test-Path -LiteralPath $Gone) { throw "Uninstall left behind $Gone" }
    }
    if ((Get-Content -Raw -LiteralPath $KeptNote) -ne 'keep-unfamiliar-file') {
        throw 'Uninstall did not preserve unfamiliar user file inside install directory'
    }
    if ((@(Get-ChildItem -LiteralPath $InstallDir -Force | ForEach-Object { $_.Name }) -join ',') -ne 'unfamiliar-user-note.txt') {
        throw 'Uninstall left something besides the unfamiliar user file in the install folder'
    }

    Assert-ProfileIntact 'after uninstall'
} finally {
    foreach ($Handle in @($ExeLock, $DatLock, $ChangingMutex, $LauncherHold)) {
        if ($Handle) { try { $Handle.Dispose() } catch { } }
    }
    foreach ($Proc in @($BusyProc, $HeldStart)) {
        if (-not $Proc) { continue }
        if (-not $Proc.HasExited) {
            try { $Proc.Kill($true); $Proc.WaitForExit(10000) | Out-Null } catch { }
        }
        try { $Proc.Dispose() } catch { }
    }
    if ($SavedUserPathCaptured) {
        $EnvKey = [Microsoft.Win32.Registry]::CurrentUser.OpenSubKey('Environment', $true)
        try {
            if ($EnvKey) {
                if ($SavedUserPathExists) {
                    $EnvKey.SetValue('Path', $SavedUserPathValue, $SavedUserPathKind)
                } elseif ($EnvKey.GetValueNames() -contains 'Path') {
                    $EnvKey.DeleteValue('Path', $false)
                }
            }
        } catch { } finally {
            if ($EnvKey) { $EnvKey.Dispose() }
        }
    }
    # Never recursively wipe $InstallDir or $UserDataDir: remove only the exact
    # files this run created, and remove a folder only when it is empty.
    if ($CreatedForeignSentinel -and (Test-Path -LiteralPath $ForeignSentinel -PathType Leaf)) {
        Remove-Item -LiteralPath $ForeignSentinel -Force -ErrorAction SilentlyContinue
    }
    if ($CreatedKeptNote -and (Test-Path -LiteralPath $KeptNote -PathType Leaf)) {
        Remove-Item -LiteralPath $KeptNote -Force -ErrorAction SilentlyContinue
    }
    if ($CreatedShortcut -and (Test-Path -LiteralPath $ShortcutPath -PathType Leaf)) {
        Remove-Item -LiteralPath $ShortcutPath -Force -ErrorAction SilentlyContinue
    }
    $TransitionNote = Join-Path $InstallDir 'transition-user-note.txt'  # step 4b's own file, if it got that far
    if ($Compiler -and (Test-Path -LiteralPath $TransitionNote -PathType Leaf)) {
        Remove-Item -LiteralPath $TransitionNote -Force -ErrorAction SilentlyContinue
    }
    if ($CreatedProfile) {
        foreach ($Rel in $ProfileFiles.Keys) {
            $Path = Join-Path $UserDataDir $Rel
            if (Test-Path -LiteralPath $Path -PathType Leaf) { Remove-Item -LiteralPath $Path -Force -ErrorAction SilentlyContinue }
        }
        foreach ($Rel in @('repo\.git', 'repo', 'data')) { Remove-DirOnlyIfEmpty (Join-Path $UserDataDir $Rel) }
        Remove-DirOnlyIfEmpty $UserDataDir
    }
    Remove-DirOnlyIfEmpty $InstallDir
    Remove-Item -LiteralPath $Work -Recurse -Force -ErrorAction SilentlyContinue
}

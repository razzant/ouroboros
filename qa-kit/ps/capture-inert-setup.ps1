# Call only after compiler_only built its local inert 1.2.3 installer, on that disposable Windows runner.
# Opens the wizard with a fresh target, reads Ready, captures that owned HWND, then closes. Never invokes Install or Finish.
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Kit,
    [Parameter(Mandatory)][string]$Setup,
    [Parameter(Mandatory)][string]$OutDir
)
Set-StrictMode -Version 3
$ErrorActionPreference = 'Stop'
if ($env:OS -ne 'Windows_NT' -or $env:OUROBOROS_QA_DISPOSABLE -ne '1' -or -not [Environment]::UserInteractive -or
    [Diagnostics.Process]::GetCurrentProcess().SessionId -eq 0 -or
    [Threading.Thread]::CurrentThread.GetApartmentState() -ne 'STA') { throw 'Requires a disposable interactive Windows desktop and Windows PowerShell -Sta.' }
$exe = Get-Item -LiteralPath $Setup
$probe = Join-Path ([IO.Path]::GetFullPath($Kit)) 'ps\brand-probe.ps1'
$OutDir = [IO.Path]::GetFullPath($OutDir)
if ($exe -isnot [IO.FileInfo] -or $exe.Name -cne 'Ouroboros-Setup-1.2.3-windows-x64.exe' -or
    ($exe.Attributes -band [IO.FileAttributes]::ReparsePoint) -or -not (Test-Path -LiteralPath $probe -PathType Leaf) -or
    (Test-Path -LiteralPath $OutDir)) { throw 'Use the compiler-only local inert output, patched kit and a fresh output directory.' }
[void](New-Item -ItemType Directory -Path $OutDir)
$capture = Join-Path $OutDir 'ready'
$destination = Join-Path $OutDir 'must-not-be-installed'

Add-Type -AssemblyName System.Drawing, UIAutomationClient, UIAutomationTypes, WindowsBase
Add-Type -Path (Join-Path $Kit 'ps\BrandNative.cs') -ReferencedAssemblies System.Drawing
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class InertSetupProbe {
    [DllImport("user32.dll")] public static extern IntPtr GetWindow(IntPtr hwnd, uint relationship);
    [DllImport("user32.dll")] private static extern uint GetWindowThreadProcessId(IntPtr hwnd, out int pid);
    [DllImport("user32.dll")] private static extern bool PostMessageW(IntPtr hwnd, uint message, IntPtr wParam, IntPtr lParam);
    public static bool Close(IntPtr hwnd, int pid) {
        int current;
        return GetWindowThreadProcessId(hwnd, out current) != 0 && current == pid &&
            PostMessageW(hwnd, 0x0010, IntPtr.Zero, IntPtr.Zero);
    }
}
'@

function Get-OwnWizard([Diagnostics.Process]$started) {
    # Always bind to the still-live process this probe started, and its own descendants; never reuse a title match from the desktop.
    if ($started.HasExited) { return }
    $pending = New-Object 'System.Collections.Generic.Queue[int]'
    $pending.Enqueue($started.Id)
    $seen = @{}
    while ($pending.Count -gt 0) {
        $id = $pending.Dequeue()
        if ($seen.ContainsKey($id)) { continue }
        if ($seen.Count -ge 16) { throw 'Unexpectedly large installer process tree.' }
        $seen[$id] = $true
        foreach ($w in [BrandQa.Native]::TopWindows($id)) {
            if ($w.Title -cmatch '^Setup - ') { $w }
        }
        foreach ($p in @(Get-CimInstance Win32_Process -Filter "ParentProcessId = $id" -Property ProcessId)) {
            $pending.Enqueue([int]$p.ProcessId)
        }
    }
}

function Get-OwnButton($w, [string]$name) {
    try {
        $ae = [System.Windows.Automation.AutomationElement]
        $root = $ae::FromHandle([IntPtr]$w.Hwnd)
        if ($root.Current.ProcessId -ne $w.Pid) { return }
        $names = [System.Windows.Automation.OrCondition]::new(
            [System.Windows.Automation.PropertyCondition]::new($ae::NameProperty, $name),
            [System.Windows.Automation.PropertyCondition]::new($ae::NameProperty, ('&' + $name)))
        foreach ($element in $root.FindAll([System.Windows.Automation.TreeScope]::Descendants, $names)) {
            $c = $element.Current
            if ($c.ProcessId -eq $w.Pid -and $c.IsEnabled -and -not $c.IsOffscreen -and -not $c.BoundingRectangle.IsEmpty -and
                ($c.ControlType -eq [System.Windows.Automation.ControlType]::Button -or $c.ClassName -match 'button')) { $element }
        }
    } catch [System.Windows.Automation.ElementNotAvailableException] { }
}

function Wait-Ready([Diagnostics.Process]$started) {
    $limit = [DateTime]::UtcNow.AddSeconds(20)
    do {
        $wins = @(Get-OwnWizard $started)
        if ($wins.Count -eq 1 -and @(Get-OwnButton $wins[0] 'Install').Count -eq 1) { return $wins[0] }
        Start-Sleep -Milliseconds 200
    } while (-not $started.HasExited -and [DateTime]::UtcNow -lt $limit)
}

function Close-OwnWizard([Diagnostics.Process]$started, $w) {
    if ($started.HasExited) { return $true }
    if ($null -eq $w) {
        $wins = @(Get-OwnWizard $started)
        if ($wins.Count -ne 1) { return $false }
        $w = $wins[0]
    }
    if (-not [InertSetupProbe]::Close([IntPtr]$w.Hwnd, $w.Pid)) { return $started.HasExited }
    $confirmed = $false
    $limit = [DateTime]::UtcNow.AddSeconds(8)
    while (-not $started.HasExited -and [DateTime]::UtcNow -lt $limit) {
        if (-not $confirmed) {
            $dialogs = @([BrandQa.Native]::TopWindows($w.Pid) | Where-Object {
                $_.Title -ceq 'Exit Setup' -and [InertSetupProbe]::GetWindow([IntPtr]$_.Hwnd, 4) -eq [IntPtr]$w.Hwnd
            })
            if ($dialogs.Count -eq 1) {
                $yes = @(Get-OwnButton $dialogs[0] 'Yes')
                if ($yes.Count -eq 1 -and $yes[0].Current.ProcessId -eq $w.Pid -and
                    [InertSetupProbe]::GetWindow([IntPtr]$dialogs[0].Hwnd, 4) -eq [IntPtr]$w.Hwnd) {
                    $invoke = $null
                    if ($yes[0].TryGetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern, [ref]$invoke)) {
                        $confirmed = $true
                        $invoke.Invoke()  # only Yes in this window's Exit Setup dialog
                    }
                }
            }
        }
        Start-Sleep -Milliseconds 100
    }
    return $started.HasExited
}

$ps = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
$started = Start-Process -FilePath $exe.FullName -ArgumentList @('/LANG=en', '/NORESTART', ('/DIR="{0}"' -f $destination)) `
    -WorkingDirectory $OutDir -PassThru
$ready = $null
$closed = $false
try {
    $ready = Wait-Ready $started
    if ($null -eq $ready) { throw 'Inert installer Ready was not observed; nothing was invoked.' }
    $p = Start-Process -FilePath $ps -NoNewWindow -PassThru -ArgumentList (
        '-NoProfile -NonInteractive -Sta -File "{0}" -Mode window -OutDir "{1}" -Label inert_ready -ProcessId {2} -Hwnd {3} -TitlePrefix "Setup - " -FramesOnly' -f
        $probe, $capture, $ready.Pid, $ready.Hwnd)
    try {
        if (-not $p.WaitForExit(15000)) {
            if (-not $p.HasExited) { $p.Kill() }  # only the exact screenshot helper just started, never Setup or a PID search
            throw 'Capture helper reached its limit; no screenshot is accepted.'
        }
        if ($p.ExitCode -ne 0) { throw 'Capture helper exited with an error.' }
    } finally { $p.Dispose() }
    if (@(Get-OwnButton $ready 'Install').Count -ne 1) { throw 'Ready was not confirmed again after drawing.' }
    $obs = Get-Content -Raw -LiteralPath (Join-Path $capture 'observations.json') | ConvertFrom-Json
    $rows = @($obs.frames | Where-Object { $_.id -ceq 'window' })
    if (-not $obs.window.found -or $obs.window.pid -ne $ready.Pid -or $obs.window.hwnd -ne $ready.Hwnd -or $rows.Count -ne 1 -or
        $rows[0].file -cne 'window-inert_ready.png' -or $rows[0].method -cnotin @('printwindow_composed', 'printwindow_default')) {
        throw 'No accepted window PNG; inspect ready/observations.json in this runner only.'
    }
    $r = $rows[0]
    $path = Join-Path $capture 'window-inert_ready.png'
    $bitmap = [Drawing.Image]::FromFile($path)
    try {
        if ($bitmap.RawFormat.Guid -ne [Drawing.Imaging.ImageFormat]::Png.Guid -or $bitmap.Width -ne $r.width -or $bitmap.Height -ne $r.height -or
            (Get-Item -LiteralPath $path).Length -ne $r.bytes -or (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ine $r.sha256) {
            throw 'Frame failed actual-file readback.'
        }
    } finally { $bitmap.Dispose() }
    Write-Host "Ready: $($r.width) x $($r.height), $($r.bytes) bytes, $($r.method), SHA256 $($r.sha256)"
} finally {
    try { $closed = Close-OwnWizard $started $ready } finally { $started.Dispose() }
    @{ ready_seen = ($null -ne $ready); setup_exit_observed = $closed; target_absent = (-not (Test-Path -LiteralPath $destination)) } |
        ConvertTo-Json -Compress | Set-Content -LiteralPath (Join-Path $OutDir 'close.json') -Encoding UTF8
}
if (-not $closed -or (Test-Path -LiteralPath $destination)) { throw 'Setup exit or absence of the installation directory could not be confirmed.' }
Write-Host 'INERT_SETUP_READY_PNG_READBACK_AND_CLOSE_OK (inspect ready/window-inert_ready.png)'

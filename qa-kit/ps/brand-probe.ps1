# Windows PowerShell 5.1, STA. Disposable interactive Windows desktop only (OUROBOROS_QA_DISPOSABLE=1).
# What Windows shows and what the shell holds for ONE app: documented APIs only (UI Automation, WM_GETICON, the shell
# property stores, PrintWindow of one addressed window, WScript.Shell). No injection, no debugging, no reading of
# another process's memory. A frame is drawn by the window itself, never cut from the screen; no pixel is judged.
#   window   : ONE window's own title/icons/properties, its frame and caption strip; -Hwnd pins the exact window. The
#              taskbar's buttons are read through UI Automation and never photographed.
#   shortcut : Target, Icon and AppUserModel.ID of the real .lnk
#   self_test: whether this host permits the above (no app needed)
# Output: <OutDir>\observations.json (RAW: may hold paths; the harness classifies it and never publishes it) + PNG frames.
[CmdletBinding()]
param(
    [Parameter(Mandatory)][ValidateSet('self_test', 'window', 'shortcut')][string]$Mode,
    [Parameter(Mandatory)][string]$OutDir,
    [ValidatePattern('^[a-z0-9_]{1,60}$')][string]$Label = 'state',
    [int]$ProcessId = 0,
    [string]$TitlePrefix = 'Ouroboros v',
    [long]$Hwnd = 0,
    [string]$ShortcutPath = '',
    [switch]$FramesOnly
)
Set-StrictMode -Version 3
$ErrorActionPreference = 'Stop'
if ($env:OS -ne 'Windows_NT' -or $env:OUROBOROS_QA_DISPOSABLE -ne '1' -or -not [Environment]::UserInteractive -or
    [Threading.Thread]::CurrentThread.GetApartmentState() -ne 'STA') {
    throw 'Use Windows PowerShell -Sta in a disposable interactive Windows session with OUROBOROS_QA_DISPOSABLE=1.'
}
$OutDir = [IO.Path]::GetFullPath($OutDir)
if (Test-Path -LiteralPath $OutDir) { throw 'OutDir must be a fresh, nonexistent directory.' }
[void](New-Item -ItemType Directory -Path $OutDir)

$FMT = '9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3'   # System.AppUserModel.* : ID=5 RelaunchCommand=2 RelaunchIconResource=3 RelaunchDisplayNameResource=4 PreventPinning=9
$obs = [ordered]@{
    schema = 1; mode = $Mode; label = $Label; errors = @(); frame_errors = [ordered]@{}; window = $null; icons = $null; props = $null
    taskbar = $null; shortcut = $null; frames = @(); host = $null; fatal = $null
}
$frames = New-Object System.Collections.ArrayList
$oldDpi = [IntPtr]::Zero

function Add-Error([string]$where, [string]$text) {
    # to the end of the quoted or unquoted run, spaces included: a profile name with a space must not survive
    $clean = [regex]::Replace($text, '(?:[A-Za-z]:\\|\\\\)[^"<>|\r\n]*', '<path>')
    $script:obs.errors = @($script:obs.errors) + ($where + ': ' + $clean)
}

function Frame-Error([string]$id, [string]$fallback, [Exception]$exception) {
    $detail = $exception.GetBaseException().Message
    $code = if ($detail -cmatch '^capture_[a-z0-9_]{1,51}$') { $detail } else { $fallback }
    $obs.frame_errors[$id] = $code
    Add-Error $id $detail  # raw paths/messages are never part of the public projection
}

function Save-Frame($bmp, [string]$id, [string]$purpose, [int[]]$rect, [string]$method) {
    if (-not [BrandQa.Native]::HasInk($bmp)) { $obs.frame_errors[$id] = 'capture_uniform_bitmap'; return $false }
    $name = $id + '-' + $Label + '.png'
    $path = Join-Path $OutDir $name
    $digest = [ExplorerQa.Native]::SaveCheckedPng($bmp, $path)
    [void]$frames.Add([ordered]@{
        id = $id; purpose = $purpose; file = $name
        sha256 = $digest; bytes = (Get-Item -LiteralPath $path).Length
        width = $bmp.Width; height = $bmp.Height; rect = $rect; method = $method
    })
    return $true
}

function Capture-OwnWindow($w, [Drawing.Rectangle]$b, [int]$dpi) {
    try {
        if ($oldDpi -eq [IntPtr]::Zero -or $dpi -le 0) { throw 'capture_dpi_unavailable' }
        if ($b.Width -lt 320 -or $b.Height -lt 240 -or $b.Width -gt 4096 -or $b.Height -gt 4096) {
            throw 'capture_unusable_window_size'
        }
        [uint32]$flags = 0
        $whole = [ExplorerQa.Native]::Capture([IntPtr]$w.Hwnd, $w.Pid, $b, $b, $true, [ref]$flags)
    } catch {
        Frame-Error 'window' 'capture_native_drawing_error' $_.Exception
        $obs.frame_errors['caption'] = $obs.frame_errors['window']
        return
    }
    try {
        $method = if ($flags -eq 2) { 'printwindow_composed' } else { 'printwindow_default' }
        try { [void](Save-Frame $whole 'window' 'the app window as it draws itself' @($b.X, $b.Y, $b.Width, $b.Height) $method) }
        catch { Frame-Error 'window' 'capture_png_save_or_decode_failed' $_.Exception }
        $capH = [int][Math]::Round(40 * ($dpi / 96.0))
        $capW = [Math]::Min($b.Width, [int][Math]::Round(640 * ($dpi / 96.0)))
        if ($capW -le 40 -or $capH -le 0 -or $capH -gt $b.Height) { $obs.frame_errors['caption'] = 'capture_caption_out_of_bounds'; return }
        try {
            $bmp = $whole.Clone([Drawing.Rectangle]::new(0, 0, $capW, $capH), [Drawing.Imaging.PixelFormat]::Format24bppRgb)
            try { [void](Save-Frame $bmp 'caption' 'the title bar: icon and name' @($b.X, $b.Y, $capW, $capH) $method) }
            finally { $bmp.Dispose() }
        } catch { Frame-Error 'caption' 'capture_png_save_or_decode_failed' $_.Exception }
    } finally { $whole.Dispose() }
}

function Read-TaskbarButtons([long]$trayHwnd) {
    $ae = [System.Windows.Automation.AutomationElement]
    $root = $ae::FromHandle([IntPtr]$trayHwnd)
    $cond = New-Object System.Windows.Automation.PropertyCondition($ae::ControlTypeProperty, [System.Windows.Automation.ControlType]::Button)
    $found = $root.FindAll([System.Windows.Automation.TreeScope]::Descendants, $cond)
    $rows = @()
    for ($i = 0; $i -lt $found.Count -and $i -lt 40; $i++) {
        $c = $found.Item($i).Current
        $r = $c.BoundingRectangle
        $rect = @(0, 0, 0, 0)
        if (-not $r.IsEmpty) { $rect = @([int][Math]::Round($r.Left), [int][Math]::Round($r.Top), [int][Math]::Round($r.Width), [int][Math]::Round($r.Height)) }
        $id = [string]$c.AutomationId
        $appid = $null
        if ($id -match '^Appid:(.+)$') { $appid = $Matches[1] }
        $rows += [ordered]@{ name = [string]$c.Name; automation_id = $id; appid = $appid; offscreen = [bool]$c.IsOffscreen; rect = $rect }
    }
    return , $rows
}

function Invoke-Taskbar {
    $tray = [BrandQa.Native]::FindTaskbar()
    $obs.taskbar = [ordered]@{ found = ($tray -ne 0); buttons = @() }
    if ($tray -eq 0) { return }
    try { $obs.taskbar.buttons = Read-TaskbarButtons $tray } catch { Add-Error 'taskbar_buttons' $_.Exception.Message }
}

function Invoke-OwnWindow($w) {
    $hwnd = [IntPtr]$w.Hwnd
    if ([ExplorerQa.Native]::WindowProcess($hwnd) -eq $w.Pid) {
        $fg = ([ExplorerQa.Native]::GetForegroundWindow() -eq $hwnd)
        $b = [ExplorerQa.Native]::WindowBounds($hwnd)
        $dpi = [int][ExplorerQa.Native]::GetDpiForWindow($hwnd)
        $obs.window.title = [string]$w.Title
        $obs.window.class_name = [string]$w.ClassName
        $obs.window.pid = [int]$w.Pid
        $obs.window.hwnd = [long]$w.Hwnd
        $obs.window.foreground = $fg
        $obs.window.dpi = $dpi
        $obs.window.rect = @($b.X, $b.Y, $b.Width, $b.Height)
        Capture-OwnWindow $w $b $dpi
        if ($FramesOnly) { return }  # transient Setup pages: do not block on Shell/branding reads before/after pixels

        # What the window itself supplies (WM_GETICON) and what Windows falls back to (class icons).
        $icons = [ordered]@{}
        foreach ($pair in @(@('small', 0), @('big', 1), @('small2', 2))) {
            $h = [BrandQa.Native]::WindowIcon([int64]$w.Hwnd, [int]$pair[1])
            $answered = ($h -ne [long]::MinValue)
            $row = [ordered]@{ window_supplies_icon = $null; bitmap_kept = $false }
            if ($answered) { $row.window_supplies_icon = ($h -ne 0) }
            if ($answered -and $h -ne 0) {
                try {
                    $bmp = [BrandQa.Native]::IconBitmap($h)
                    try { $row.bitmap_kept = [bool](Save-Frame $bmp ('own-icon-' + $pair[0]) 'the icon bitmap the window itself supplies' @(0, 0, $bmp.Width, $bmp.Height) 'window_icon') }
                    finally { $bmp.Dispose() }
                } catch { Frame-Error ('own-icon-' + $pair[0]) 'capture_icon_bitmap_error' $_.Exception }
            }
            $icons[$pair[0]] = $row
        }
        foreach ($pair in @(@('class_small', $true), @('class_big', $false))) {
            $h = [BrandQa.Native]::ClassIcon([int64]$w.Hwnd, [bool]$pair[1])
            $row = [ordered]@{ class_has_icon = $null }  # a failed read stays null (unread), never "no icon"
            if ($h -ne [long]::MinValue) { $row.class_has_icon = ($h -ne 0) }
            $icons[$pair[0]] = $row
        }
        $obs.icons = $icons

        # The window's own AppUserModel properties (null = not set at window level; a process-wide ID is not readable here).
        $props = [ordered]@{}
        foreach ($pair in @(@('aumid', 5), @('relaunch_command', 2), @('relaunch_icon_resource', 3), @('relaunch_display_name_resource', 4), @('prevent_pinning', 9))) {
            try { $props[$pair[0]] = [BrandQa.Native]::WindowProp([int64]$w.Hwnd, $FMT, [uint32]$pair[1]) } catch { Add-Error ('prop_' + $pair[0]) $_.Exception.Message }
        }
        $obs.props = $props
    } else { throw 'capture_window_identity_or_visibility_changed' }
}

function Invoke-WindowMode {
    $wins = @()
    if ($ProcessId -gt 0) {
        $wins = @([BrandQa.Native]::TopWindows($ProcessId) | Where-Object {
            $_.Title.StartsWith($TitlePrefix, [StringComparison]::OrdinalIgnoreCase) -and ($Hwnd -eq 0 -or $_.Hwnd -eq $Hwnd) })
    }
    $obs.window = [ordered]@{ found = ($wins.Count -eq 1); candidates = $wins.Count }
    if ($wins.Count -eq 1) {
        try { Invoke-OwnWindow $wins[0] }
        catch {
            Frame-Error 'window' 'capture_window_probe_error' $_.Exception
            $obs.frame_errors['caption'] = $obs.frame_errors['window']
        }
    }
    if (-not $FramesOnly) {
        try { Invoke-Taskbar } catch { Add-Error 'taskbar' $_.Exception.Message }
    }
}

function Invoke-ShortcutMode {
    $sh = New-Object -ComObject WScript.Shell
    $lnk = $sh.CreateShortcut($ShortcutPath)
    $obs.shortcut = [ordered]@{
        exists = (Test-Path -LiteralPath $ShortcutPath -PathType Leaf)
        target = [string]$lnk.TargetPath; arguments = [string]$lnk.Arguments; working_directory = [string]$lnk.WorkingDirectory
        icon_location = [string]$lnk.IconLocation; description = [string]$lnk.Description
        aumid = [BrandQa.Native]::FileProp($ShortcutPath, $FMT, 5)
    }
}

function Invoke-SelfTest {
    $virtual = [System.Windows.Forms.SystemInformation]::VirtualScreen
    $tray = [BrandQa.Native]::FindTaskbar()
    $facts = [ordered]@{
        powershell = $PSVersionTable.PSVersion.ToString(); session_id = [Diagnostics.Process]::GetCurrentProcess().SessionId
        foreground_present = ([ExplorerQa.Native]::GetForegroundWindow() -ne [IntPtr]::Zero)
        screen = @($virtual.Width, $virtual.Height); taskbar_found = ($tray -ne 0)
        taskbar_rect = [BrandQa.Native]::WindowRectArray($tray); taskbar_button_count = -1
        shell_application_com = $false; wscript_shell_com = $false
    }
    try { $facts.taskbar_button_count = @(Read-TaskbarButtons $tray).Count } catch { Add-Error 'taskbar_buttons' $_.Exception.Message }
    try { [void](New-Object -ComObject Shell.Application); $facts.shell_application_com = $true } catch { Add-Error 'shell_application' $_.Exception.Message }
    try { [void](New-Object -ComObject WScript.Shell); $facts.wscript_shell_com = $true } catch { Add-Error 'wscript_shell' $_.Exception.Message }
    $obs.host = $facts
}

try {
    Add-Type -AssemblyName System.Drawing, System.Windows.Forms, UIAutomationClient, UIAutomationTypes, WindowsBase
    Add-Type -Path (Join-Path $PSScriptRoot 'ExplorerCapture.cs') -ReferencedAssemblies System.Drawing
    Add-Type -Path (Join-Path $PSScriptRoot 'BrandNative.cs') -ReferencedAssemblies System.Drawing
    $oldDpi = [ExplorerQa.Native]::SetThreadDpiAwarenessContext([IntPtr](-4))   # per-monitor V2: physical pixels
    if ($oldDpi -eq [IntPtr]::Zero) { Add-Error 'dpi' 'could not bind the thread to physical pixels' }
    if ($Mode -eq 'window') { Invoke-WindowMode }
    elseif ($Mode -eq 'shortcut') { Invoke-ShortcutMode }
    else { Invoke-SelfTest }
}
catch {
    $obs.fatal = [regex]::Replace($_.Exception.Message, '(?:[A-Za-z]:\\|\\\\)[^"<>|\r\n]*', '<path>')
    throw
}
finally {
    $obs.frames = $frames
    $obs | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $OutDir 'observations.json') -Encoding UTF8
    if ($oldDpi -ne [IntPtr]::Zero) { [void][ExplorerQa.Native]::SetThreadDpiAwarenessContext($oldDpi) }
}

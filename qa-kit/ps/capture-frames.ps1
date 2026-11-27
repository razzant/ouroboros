# Windows PowerShell 5.1, STA. Disposable interactive Windows desktop only (OUROBOROS_QA_DISPOSABLE=1).
# Adapted from explorer-qa/capture-explorer.ps1 with the same capture rules (a NEW Explorer window of our own, the
# Details view, an on-screen UI Automation row that must be unique, matched geometry across frames, no selection; the
# PNG is that row only, drawn by the window itself, never cut from the screen). What changed: the frames come from a JSON spec instead of one downloaded release ZIP,
# and one frame's failure no longer aborts the others. The local C# helper is adapted from explorer-qa/ExplorerCapture.cs.
# If Explorer cannot deliver a row, an opted-in exact Shell item may supply just its native file icon. It is a distinct surface.
# Nothing here runs the app, and nothing judges the artwork: a captured frame is "pending_visual_inspection".
#
# Spec: {"stage": "...", "frames": [{"id": "...", "folder": "<folder shown>", "folder_label": "...", "item": "<file name>", "shell_item_icon": true}]}
# Output: <OutDir>\observations.json (path-free) + <id>.png for a row, or shell_item_<id>.png for a file icon.
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$SpecFile,
    [Parameter(Mandatory)][string]$OutDir
)
Set-StrictMode -Version 3
$ErrorActionPreference = 'Stop'
if ($env:OS -ne 'Windows_NT' -or $env:OUROBOROS_QA_DISPOSABLE -ne '1' -or
    -not [Environment]::UserInteractive -or
    [Threading.Thread]::CurrentThread.GetApartmentState() -ne 'STA') {
    throw 'Use Windows PowerShell -Sta in a disposable interactive Windows session with OUROBOROS_QA_DISPOSABLE=1.'
}
$OutDir = [IO.Path]::GetFullPath($OutDir)
if (Test-Path -LiteralPath $OutDir) { throw 'OutDir must be a fresh, nonexistent directory.' }
[void](New-Item -ItemType Directory -Path $OutDir)
$spec = Get-Content -LiteralPath $SpecFile -Raw -Encoding UTF8 | ConvertFrom-Json
$specFrames = @($spec.frames)

$report = [ordered]@{
    schema = 1; stage = [string]$spec.stage; host = [ordered]@{}; frames = [ordered]@{}; fatal = $null
    visual = 'pending_visual_inspection'; app_executed = $false
    method = 'Windows Explorer Shell.Application + on-screen UI Automation row + HWND-bound PrintWindow, cut to the target row'
}
$script:firstWindow = $null
$win = $null
$hwnd = [IntPtr]::Zero
$explorerProcess = $null
$oldDpi = [IntPtr]::Zero

function Scrub([string]$text) { return [regex]::Replace($text, '(?:[A-Za-z]:\\|\\\\)[^"<>|\r\n]*', '<path>') }

# Explorer's Document, and its Folder, are empty (nullable) while it is still navigating: '' then, never an error under StrictMode.
function Get-ViewPath($view) {
    if ($null -eq $view) { return '' }
    $shown = $view.Folder
    if ($null -eq $shown -or $null -eq $shown.Self) { return '' }
    return [string]$shown.Self.Path
}

function Wait-For($probe, [string]$failure) {
    $limit = [Diagnostics.Stopwatch]::StartNew()
    do {
        try { $value = & $probe; if ($null -ne $value) { return $value } }
        catch [Runtime.InteropServices.COMException] { }  # Explorer can change views during a read.
        catch [System.Windows.Automation.ElementNotAvailableException] { }
        Start-Sleep -Milliseconds 250
    } while ($limit.Elapsed.TotalSeconds -lt 25)
    throw $failure
}

function Get-ExplorerWindows {
    $windows = $shell.Windows()
    for ($i = 0; $i -lt $windows.Count; $i++) {
        try {
            $w = $windows.Item($i)
            if ($null -ne $w -and [IO.Path]::GetFileName($w.FullName) -ieq 'explorer.exe') { $w }
        } catch [Runtime.InteropServices.COMException] { }
    }
}

function Save-ShellItemIcon($f, $facts) {
    if ($null -eq $f.PSObject.Properties['shell_item_icon'] -or $f.shell_item_icon -ne $true) { return }
    $label = [string]$f.id
    $itemName = [string]$f.item
    if ($label -notmatch '^explorer_[a-z0-9_]+$' -or $itemName -ne [IO.Path]::GetFileName($itemName) -or
        [IO.Path]::GetExtension($itemName) -notin @('.exe', '.lnk')) { return }
    $folder = $shell.NameSpace([string]$f.folder)
    if ($null -eq $folder -or $folder.Self.Path -ine [string]$f.folder) { return }
    $item = $folder.ParseName($itemName)
    $expected = Join-Path ([string]$f.folder) $itemName
    if ($null -eq $item -or $item.IsFolder -or $item.Path -ine $expected) { return }
    $file = Get-Item -LiteralPath $expected
    if ($file -isnot [IO.FileInfo] -or ($file.Attributes -band [IO.FileAttributes]::ReparsePoint)) { return }
    if (-not ('BrandQa.Native' -as [type])) {
        Add-Type -Path (Join-Path $PSScriptRoot 'BrandNative.cs') -ReferencedAssemblies System.Drawing
    }
    $bitmap = [BrandQa.Native]::ShellItemIcon($file.FullName)
    if ($null -eq $bitmap) { return }
    try {
        if (-not [BrandQa.Native]::HasInk($bitmap)) { return }
        $name = "shell_item_$label.png"
        $png = Join-Path $OutDir $name
        $digest = [ExplorerQa.Native]::SaveCheckedPng($bitmap, $png)
        $facts.png = [ordered]@{ file = $name; sha256 = $digest; bytes = (Get-Item -LiteralPath $png).Length
            width = $bitmap.Width; height = $bitmap.Height; method = 'shell_item_icon' }
        $facts.status = 'icon_only'
    } finally { $bitmap.Dispose() }
}

function Read-Frame($f) {
    $label = [string]$f.id
    $folder = [string]$f.folder
    $itemName = [string]$f.item
    $facts = [ordered]@{
        folder_label = [string]$f.folder_label; stage = 'navigate'; status = 'failed'
        error = $null; capture_reason = $null; shell = $null; row = $null; png = $null; window = $null
    }
    $report.frames[$label] = $facts
    try {
        if ($null -eq $win) { throw 'No owned Explorer window exists; nothing is captured.' }
        $win.Navigate2($folder)
        $view = Wait-For {
            if (-not $win.Busy -and (Get-ViewPath $win.Document) -ieq $folder) { $win.Document }
        } "Explorer did not navigate to the required folder (a ZIP namespace may be unavailable)."

        $facts.stage = 'shell_item'
        $item = $view.Folder.ParseName($itemName)
        if ($null -eq $item -or $item.IsFolder -or $item.Path -ine (Join-Path $folder $itemName)) {
            throw 'No exact Shell file for the item; refusing to capture a substitute row.'
        }
        # Details = FVM_DETAILS (4). Re-read on the active view, after each navigation.
        $view.CurrentViewMode = 4
        $mode = Wait-For {
            $d = $win.Document
            if (-not $win.Busy -and (Get-ViewPath $d) -ieq $folder -and [int]$d.CurrentViewMode -eq 4) { 4 }
        } 'Explorer did not accept Details view.'
        $facts.shell = [ordered]@{ item_display_name = $item.Name; item_type = $item.Type; view = $mode }

        $facts.stage = 'window_and_row'
        $bounds = [ExplorerQa.Native]::WindowBounds($hwnd)
        $dpi = [ExplorerQa.Native]::GetDpiForWindow($hwnd)
        $monitor = [Windows.Forms.Screen]::FromHandle($hwnd).Bounds
        if ($dpi -eq 0 -or $bounds.Width -lt 480 -or $bounds.Height -lt 340 -or -not $monitor.Contains($bounds)) {
            throw 'Target must be unminimized, large enough and fully within one monitor.'
        }
        $facts.window = [ordered]@{ dpi = $dpi; rect = @($bounds.X, $bounds.Y, $bounds.Width, $bounds.Height)
            monitor_rect = @($monitor.X, $monitor.Y, $monitor.Width, $monitor.Height) }
        if ($null -eq $script:firstWindow) { $script:firstWindow = $facts.window }
        else {
            $first = $script:firstWindow
            if ($dpi -ne $first.dpi -or ($facts.window.rect -join ',') -ne ($first.rect -join ',') -or
                ($facts.window.monitor_rect -join ',') -ne ($first.monitor_rect -join ',')) {
                throw 'Window geometry, display or DPI changed between frames.'
            }
        }

        $stem = [IO.Path]::GetFileNameWithoutExtension($itemName)
        $ae = [System.Windows.Automation.AutomationElement]
        $ct = [System.Windows.Automation.ControlType]
        $nameCond = [System.Windows.Automation.PropertyCondition]::new($ae::NameProperty, $itemName)
        if ($stem -ne $itemName) {
            $nameCond = [System.Windows.Automation.OrCondition]::new($nameCond,
                [System.Windows.Automation.PropertyCondition]::new($ae::NameProperty, $stem))
        }
        $typeCond = [System.Windows.Automation.OrCondition]::new(
            [System.Windows.Automation.PropertyCondition]::new($ae::ControlTypeProperty, $ct::ListItem),
            [System.Windows.Automation.PropertyCondition]::new($ae::ControlTypeProperty, $ct::DataItem))
        $namedItem = [System.Windows.Automation.AndCondition]::new($nameCond, $typeCond)

        $row = Wait-For {
            $root = [System.Windows.Automation.AutomationElement]::FromHandle($hwnd)
            $matches = @($root.FindAll([System.Windows.Automation.TreeScope]::Descendants, $namedItem) |
                Where-Object {
                    $c = $_.Current
                    -not $c.IsOffscreen -and -not $c.BoundingRectangle.IsEmpty -and
                    $c.ProcessId -eq $explorerProcess.Id -and $c.BoundingRectangle.Width -gt 0 -and
                    $c.BoundingRectangle.Height -gt 0 -and $c.BoundingRectangle.Left -ge $bounds.Left -and
                    $c.BoundingRectangle.Right -le $bounds.Right -and $c.BoundingRectangle.Top -ge $bounds.Top -and
                    $c.BoundingRectangle.Bottom -le $bounds.Bottom
                })
            if ($matches.Count -eq 1) { $matches[0] }
        } 'No unique on-screen row exposed by Explorer UI Automation.'
        $current = $row.Current
        $r = $current.BoundingRectangle
        $rowPixels = [Drawing.Rectangle]::FromLTRB(
            [int][Math]::Ceiling($r.Left - $bounds.Left), [int][Math]::Ceiling($r.Top - $bounds.Top),
            [int][Math]::Floor($r.Right - $bounds.Left), [int][Math]::Floor($r.Bottom - $bounds.Top))
        $facts.row = [ordered]@{ name = $current.Name; control = $current.ControlType.ProgrammaticName
            rect_in_window = @($rowPixels.X, $rowPixels.Y, $rowPixels.Width, $rowPixels.Height) }
        $facts.stage = 'capture'
        # Re-bind immediately before pixels: tab changes and temporary overlays are not the test.
        $facts.capture_reason = 'capture_explorer_view_changed_before_pixels'
        if ($win.Busy -or (Get-ViewPath $win.Document) -ine $folder -or [int]$win.Document.CurrentViewMode -ne 4 -or
            [ExplorerQa.Native]::WindowBounds($hwnd) -ne $bounds -or
            [ExplorerQa.Native]::GetDpiForWindow($hwnd) -ne $dpi -or
            [int]$win.Document.SelectedItems().Count -ne 0) { throw 'Folder/view/geometry changed or selection is nonempty.' }
        $facts.capture_reason = 'capture_explorer_native_drawing_error'
        [uint32]$flags = 0
        $part = [Drawing.Rectangle]::new($bounds.X + $rowPixels.X, $bounds.Y + $rowPixels.Y, $rowPixels.Width, $rowPixels.Height)
        $bitmap = [ExplorerQa.Native]::Capture($hwnd, $explorerProcess.Id, $bounds, $part, $false, [ref]$flags)
        try {
            $facts.capture_reason = 'capture_explorer_view_changed_after_pixels'
            if ($win.Busy -or (Get-ViewPath $win.Document) -ine $folder -or [int]$win.Document.CurrentViewMode -ne 4 -or
                [ExplorerQa.Native]::GetDpiForWindow($hwnd) -ne $dpi -or
                [int]$win.Document.SelectedItems().Count -ne 0 -or $row.Current.IsOffscreen -or $row.Current.BoundingRectangle -ne $r) {
                throw 'Folder, DPI, selection or the target row changed during capture.'
            }
            $name = "$label.png"
            $png = Join-Path $OutDir $name
            $facts.capture_reason = 'capture_png_save_or_decode_failed'
            $digest = [ExplorerQa.Native]::SaveCheckedPng($bitmap, $png)
            $method = if ($flags -eq 2) { 'printwindow_composed' } else { 'printwindow_default' }
            $facts.png = [ordered]@{ file = $name; sha256 = $digest; bytes = (Get-Item -LiteralPath $png).Length
                width = $bitmap.Width; height = $bitmap.Height; method = $method }
        } finally { $bitmap.Dispose() }
        $facts.stage = 'captured_pending_visual_inspection'
        $facts.status = 'captured'
        $facts.capture_reason = $null
    }
    catch {
        $detail = $_.Exception.GetBaseException().Message
        if ($detail -cmatch '^capture_[a-z0-9_]{1,51}$') { $facts.capture_reason = $detail }
        $facts.error = Scrub $_.Exception.Message
        try { Save-ShellItemIcon $f $facts } catch { $facts.icon_error = Scrub $_.Exception.Message }
    }
}

try {
    $os = Get-CimInstance -ClassName Win32_OperatingSystem
    $report.host = [ordered]@{
        windows = $os.Caption; version = $os.Version; build = $os.BuildNumber
        architecture = $os.OSArchitecture; powershell = $PSVersionTable.PSVersion.ToString()
        session_id = [Diagnostics.Process]::GetCurrentProcess().SessionId
        runner_image = $env:ImageOS; runner_image_version = $env:ImageVersion
    }
    if ($report.host.session_id -eq 0) { throw 'Session 0 has no interactive Explorer surface.' }
    Add-Type -AssemblyName System.Drawing, System.Windows.Forms, UIAutomationClient, UIAutomationTypes, WindowsBase
    Add-Type -Path (Join-Path $PSScriptRoot 'ExplorerCapture.cs') -ReferencedAssemblies System.Drawing
    $oldDpi = [ExplorerQa.Native]::SetThreadDpiAwarenessContext([IntPtr](-4)) # Per-monitor V2.
    if ($oldDpi -eq [IntPtr]::Zero) { throw 'Failed to bind capture thread to physical pixels.' }
    $shell = New-Object -ComObject Shell.Application
    $firstFolder = [string]$specFrames[0].folder
    $before = @(Get-ExplorerWindows | ForEach-Object { [Int64]$_.HWND })
    $shell.Explore($firstFolder)
    try {
        $win = Wait-For {
            $opened = @(Get-ExplorerWindows | Where-Object {
                [Int64]$_.HWND -notin $before -and -not $_.Busy -and (Get-ViewPath $_.Document) -ieq $firstFolder
            })
            if ($opened.Count -eq 1) { $opened[0] }
        } 'No identifiable new Explorer window was created. No other window will be reused or captured.'
        $hwnd = [IntPtr]([Int64]$win.HWND)
        $explorerProcess = [Diagnostics.Process]::GetProcessById([ExplorerQa.Native]::WindowProcess($hwnd))
        if ($explorerProcess.ProcessName -ine 'explorer' -or $explorerProcess.SessionId -ne $report.host.session_id) {
            throw 'Shell window is not Explorer in this interactive session.'
        }
        $report.host.explorer = [ordered]@{ name = $explorerProcess.ProcessName
            binary_version = (Get-Item -LiteralPath (Join-Path $env:SystemRoot 'explorer.exe')).VersionInfo.FileVersion }
    } catch {
        $report.fatal = Scrub $_.Exception.Message
        $win = $null
    }
    foreach ($f in $specFrames) { Read-Frame $f }
}
catch {
    $report.fatal = Scrub $_.Exception.Message
    throw
}
finally {
    $report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $OutDir 'observations.json') -Encoding UTF8
    if ($oldDpi -ne [IntPtr]::Zero) { [void][ExplorerQa.Native]::SetThreadDpiAwarenessContext($oldDpi) }
    if ($null -ne $win) { try { $win.Quit() } catch { Write-Warning 'Could not close the owned Explorer window; runner disposal will close it.' } }
}

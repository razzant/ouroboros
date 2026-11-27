# Windows PowerShell 5.1, STA. Disposable interactive Windows desktop only (OUROBOROS_QA_DISPOSABLE=1).
# Adapted from explorer-qa/capture-explorer.ps1 with the same capture rules (a NEW Explorer window of our own, the
# Details view, an on-screen UI Automation row that must be unique, matched geometry across frames, no selection; the
# PNG is that row only, drawn by the window itself, never cut from the screen). What changed: the frames come from a JSON spec instead of one downloaded release ZIP,
# and one frame's failure no longer aborts the others. The shared C# helper is explorer-qa's ExplorerCapture.cs, byte for byte.
# Nothing here runs the app, and nothing judges a pixel: a captured frame is "pending_visual_inspection".
#
# Spec: {"stage": "...", "frames": [{"id": "...", "folder": "<folder shown>", "folder_label": "...", "item": "<file name>"}]}
# Output: <OutDir>\observations.json (path-free) + <id>.png per captured frame.
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
    method = 'Windows Explorer Shell.Application + on-screen UI Automation row + PrintWindow of that window, cut to the target row'
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

function Read-Frame($f) {
    $label = [string]$f.id
    $folder = [string]$f.folder
    $itemName = [string]$f.item
    $facts = [ordered]@{
        folder_label = [string]$f.folder_label; stage = 'navigate'; status = 'failed'
        error = $null; shell = $null; row = $null; png = $null; window = $null
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
        if ((Get-ViewPath $win.Document) -ine $folder -or [int]$win.Document.CurrentViewMode -ne 4 -or
            [ExplorerQa.Native]::WindowBounds($hwnd) -ne $bounds -or
            [ExplorerQa.Native]::GetDpiForWindow($hwnd) -ne $dpi -or
            [int]$win.Document.SelectedItems().Count -ne 0) { throw 'Folder/view/geometry changed or selection is nonempty.' }
        $bitmap = [ExplorerQa.Native]::Capture($hwnd, $bounds, $rowPixels)
        try {
            if ($win.Busy -or (Get-ViewPath $win.Document) -ine $folder -or
                -not [ExplorerQa.Native]::HasRowInk($bitmap, [Drawing.Rectangle]::new(0, 0, $bitmap.Width, $bitmap.Height))) { throw 'Folder changed or the row is blank.' }
            $name = "$label.png"
            $png = Join-Path $OutDir $name
            $bitmap.Save($png, [Drawing.Imaging.ImageFormat]::Png)
            $facts.png = [ordered]@{ file = $name; sha256 = (Get-FileHash -LiteralPath $png -Algorithm SHA256).Hash.ToLowerInvariant() }
        } finally { $bitmap.Dispose() }
        $facts.stage = 'captured_pending_visual_inspection'
        $facts.status = 'captured'
    }
    catch {
        $facts.error = Scrub $_.Exception.Message
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

# Windows PowerShell 5.1, STA. Disposable interactive Windows desktop only.
[CmdletBinding()]
param([Parameter(Mandatory)][string]$RunRoot)
Set-StrictMode -Version 3
$ErrorActionPreference = 'Stop'
if ($env:OS -ne 'Windows_NT' -or $env:OUROBOROS_EXPLORER_QA -ne '1' -or
    -not [Environment]::UserInteractive -or
    [Threading.Thread]::CurrentThread.GetApartmentState() -ne 'STA') {
    throw 'Use Windows PowerShell -Sta in a disposable interactive Windows session with OUROBOROS_EXPLORER_QA=1.'
}

$RunRoot = [IO.Path]::GetFullPath($RunRoot)
$asset = 'Ouroboros-7.12.0-windows-x64.zip'
$member = 'Ouroboros/Ouroboros.exe'
$expectedZip = 'bd92f35119f86da8361a3aa43f4e694ff7e7f79450219370cbcec0a93832f98d'
$expectedExe = '3cf8c391bb65540a2683d2391afe580cd6d29d81b3e9d853e5f2e62ad7a101a0'
$url = 'https://github.com/razzant/ouroboros/releases/download/v7.12.0/' + $asset
# Never reuse an evidence directory: a failed capture must not inherit an old PNG.
$evidence = Join-Path $RunRoot 'evidence'
if (Test-Path -LiteralPath $RunRoot) { throw 'RunRoot must be a fresh, nonexistent directory.' }
[void](New-Item -ItemType Directory -Path $RunRoot)
[void](New-Item -ItemType Directory -Path $evidence)
$zip = Join-Path $RunRoot $asset
$extracted = Join-Path $RunRoot 'extracted'
$report = [ordered]@{
    schema = 1; source = [ordered]@{
        url = $url; member = $member; zip_expected = $expectedZip; exe_expected = $expectedExe
    }; host = [ordered]@{}; frames = [ordered]@{}; visual = 'unconfirmed'; stage = 'host'
    method = 'Windows Explorer Shell.Application + on-screen UI Automation row + PrintWindow of that window, cut to the target row'
    app_executed = $false
}

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

function Read-Frame([string]$label, [string]$folder) {
    $facts = [ordered]@{ stage = 'navigate'; shell = $null; row = $null; png = $null }
    $report.frames[$label] = $facts
    $win.Navigate2($folder)
    $view = Wait-For {
        if (-not $win.Busy -and (Get-ViewPath $win.Document) -ieq $folder) { $win.Document }
    } "Explorer did not navigate to the required $label folder (ZIP namespace may be unavailable)."

    $facts.stage = 'shell_item'
    $item = $view.Folder.ParseName('Ouroboros.exe')
    if ($null -eq $item -or $item.IsFolder -or $item.Path -ine (Join-Path $folder 'Ouroboros.exe')) {
        throw "No exact Shell file in $label; refusing to capture a substitute row."
    }
    # Details = FVM_DETAILS (4). Re-read on the active view, after each navigation.
    $view.CurrentViewMode = 4
    $mode = Wait-For {
        $d = $win.Document
        if (-not $win.Busy -and (Get-ViewPath $d) -ieq $folder -and [int]$d.CurrentViewMode -eq 4) { 4 }
    } "Explorer did not accept Details view in $label."
    $facts.shell = [ordered]@{
        folder_relative = $folder.Substring($RunRoot.Length).TrimStart('\')
        item_relative = $item.Path.Substring($RunRoot.Length).TrimStart('\')
        item_display_name = $item.Name; item_type = $item.Type; view = $mode
    }

    $facts.stage = 'window_and_row'
    $bounds = [ExplorerQa.Native]::WindowBounds($hwnd)
    $dpi = [ExplorerQa.Native]::GetDpiForWindow($hwnd)
    $monitor = [Windows.Forms.Screen]::FromHandle($hwnd).Bounds
    if ($dpi -eq 0 -or $bounds.Width -lt 480 -or $bounds.Height -lt 340 -or -not $monitor.Contains($bounds)) {
        throw "Target must be unminimized, large enough and fully within one monitor in $label."
    }
    $facts.window = [ordered]@{ hwnd = $hwnd.ToInt64(); pid = $explorerProcess.Id; dpi = $dpi
        rect = @($bounds.X,$bounds.Y,$bounds.Width,$bounds.Height)
        monitor_rect = @($monitor.X,$monitor.Y,$monitor.Width,$monitor.Height) }
    if ($label -eq 'extracted') {
        $first = $report.frames['in-zip'].window
        if ($dpi -ne $first.dpi -or ($facts.window.rect -join ',') -ne ($first.rect -join ',') -or
            ($facts.window.monitor_rect -join ',') -ne ($first.monitor_rect -join ',')) {
            throw 'Window geometry, display or DPI changed between frames.'
        }
    }

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
    } "No unique on-screen Ouroboros row exposed by Explorer UI Automation in $label."
    $current = $row.Current
    $r = $current.BoundingRectangle
    $rowPixels = [Drawing.Rectangle]::FromLTRB(
        [int][Math]::Ceiling($r.Left - $bounds.Left), [int][Math]::Ceiling($r.Top - $bounds.Top),
        [int][Math]::Floor($r.Right - $bounds.Left), [int][Math]::Floor($r.Bottom - $bounds.Top))
    $facts.row = [ordered]@{ name = $current.Name; control = $current.ControlType.ProgrammaticName
        rect_in_window = @($rowPixels.X,$rowPixels.Y,$rowPixels.Width,$rowPixels.Height) }
    $facts.stage = 'capture'
    # Re-bind immediately before pixels: tab changes and temporary overlays are not the test.
    if ((Get-ViewPath $win.Document) -ine $folder -or [int]$win.Document.CurrentViewMode -ne 4 -or
        [ExplorerQa.Native]::WindowBounds($hwnd) -ne $bounds -or
        [ExplorerQa.Native]::GetDpiForWindow($hwnd) -ne $dpi -or
        [int]$win.Document.SelectedItems().Count -ne 0) { throw "Folder/view/geometry changed or selection is nonempty in $label." }
    $bitmap = [ExplorerQa.Native]::Capture($hwnd, $bounds, $rowPixels)
    try {
        if ($win.Busy -or (Get-ViewPath $win.Document) -ine $folder -or
            -not [ExplorerQa.Native]::HasRowInk($bitmap, [Drawing.Rectangle]::new(0, 0, $bitmap.Width, $bitmap.Height))) { throw "Folder changed or the row is blank in $label." }
        $name = "$label.png"
        $png = Join-Path $evidence $name
        $bitmap.Save($png, [Drawing.Imaging.ImageFormat]::Png)
        $facts.png = [ordered]@{ file = $name; sha256 = (Get-FileHash -LiteralPath $png -Algorithm SHA256).Hash.ToLowerInvariant() }
    } finally { $bitmap.Dispose() }
    $facts.stage = 'captured_pending_visual_inspection'
}

$win = $null
$oldDpi = [IntPtr]::Zero
try {
    $os = Get-CimInstance -ClassName Win32_OperatingSystem
    $report.host = [ordered]@{
        windows = $os.Caption; version = $os.Version; build = $os.BuildNumber
        architecture = $os.OSArchitecture; powershell = $PSVersionTable.PSVersion.ToString()
        session_id = [Diagnostics.Process]::GetCurrentProcess().SessionId
        runner_image = $env:ImageOS; runner_image_version = $env:ImageVersion
    }
    if ($report.host.session_id -eq 0) { throw 'Session 0 has no interactive Explorer surface.' }

    $report.stage = 'source'
    & curl.exe --fail --location --silent --show-error --max-time 480 --output $zip $url
    if ($LASTEXITCODE -ne 0) { throw "Public ZIP download failed (curl exit $LASTEXITCODE)." }
    $report.source.zip_bytes = (Get-Item -LiteralPath $zip).Length
    $report.source.zip_actual = (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($report.source.zip_actual -ne $expectedZip) { throw 'Full archive SHA-256 mismatch.' }
    $report.source.zip_has_zone_identifier = $null -ne (Get-Item -LiteralPath $zip -Stream Zone.Identifier -ErrorAction SilentlyContinue)
    Add-Type -AssemblyName System.IO.Compression, System.IO.Compression.FileSystem
    [IO.Compression.ZipFile]::ExtractToDirectory($zip, $extracted)
    $diskFolder = Join-Path $extracted 'Ouroboros'
    $exe = Join-Path $diskFolder 'Ouroboros.exe'
    $report.source.exe_actual = (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($report.source.exe_actual -ne $expectedExe) { throw 'Extracted executable SHA-256 mismatch.' }
    $report.source.exe_has_zone_identifier = $null -ne (Get-Item -LiteralPath $exe -Stream Zone.Identifier -ErrorAction SilentlyContinue)

    $report.stage = 'desktop'
    Add-Type -AssemblyName System.Drawing, System.Windows.Forms, UIAutomationClient, UIAutomationTypes, WindowsBase
    Add-Type -Path (Join-Path $PSScriptRoot 'ExplorerCapture.cs') -ReferencedAssemblies System.Drawing
    $oldDpi = [ExplorerQa.Native]::SetThreadDpiAwarenessContext([IntPtr](-4)) # Per-monitor V2.
    if ($oldDpi -eq [IntPtr]::Zero) { throw 'Failed to bind capture thread to physical pixels.' }
    $ae = [System.Windows.Automation.AutomationElement]
    $ct = [System.Windows.Automation.ControlType]
    $namedItem = [System.Windows.Automation.AndCondition]::new(
        [System.Windows.Automation.OrCondition]::new(
            [System.Windows.Automation.PropertyCondition]::new($ae::NameProperty, 'Ouroboros.exe'),
            [System.Windows.Automation.PropertyCondition]::new($ae::NameProperty, 'Ouroboros')),
        [System.Windows.Automation.OrCondition]::new(
            [System.Windows.Automation.PropertyCondition]::new($ae::ControlTypeProperty, $ct::ListItem),
            [System.Windows.Automation.PropertyCondition]::new($ae::ControlTypeProperty, $ct::DataItem)))
    $shell = New-Object -ComObject Shell.Application
    $before = @(Get-ExplorerWindows | ForEach-Object { [Int64]$_.HWND })
    $shell.Explore($RunRoot)
    $win = Wait-For {
        $opened = @(Get-ExplorerWindows | Where-Object {
            [Int64]$_.HWND -notin $before -and -not $_.Busy -and (Get-ViewPath $_.Document) -ieq $RunRoot
        })
        if ($opened.Count -eq 1) { $opened[0] }
    } 'No identifiable new Explorer window was created. No other window will be reused or captured.'
    $hwnd = [IntPtr]([Int64]$win.HWND)
    $explorerProcess = [Diagnostics.Process]::GetProcessById([ExplorerQa.Native]::WindowProcess($hwnd))
    if ($explorerProcess.ProcessName -ine 'explorer' -or $explorerProcess.SessionId -ne $report.host.session_id) {
        throw 'Shell window is not Explorer in this interactive session.'
    }
    $report.host.explorer = [ordered]@{ name = $explorerProcess.ProcessName; pid = $explorerProcess.Id
        binary_version = (Get-Item -LiteralPath (Join-Path $env:SystemRoot 'explorer.exe')).VersionInfo.FileVersion }
    $report.stage = 'in-zip'
    Read-Frame 'in-zip' ($zip + '\Ouroboros')
    $report.stage = 'extracted'
    Read-Frame 'extracted' $diskFolder
    $report.stage = 'capture_complete'
    $report.visual = 'pending_visual_inspection'
    Write-Host 'Both Explorer frames captured; icon appearance still requires visual inspection.'
} catch {
    $report.visual = 'failed_to_capture_confirmed_pair'
    # the published report keeps the exception type only ($report.stage says where); the message stays in the job log
    $report.error = $_.Exception.GetType().Name
    throw
} finally {
    $report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $evidence 'observations.json') -Encoding UTF8
    if ($oldDpi -ne [IntPtr]::Zero) { [void][ExplorerQa.Native]::SetThreadDpiAwarenessContext($oldDpi) }
    if ($null -ne $win) { try { $win.Quit() } catch { Write-Warning 'Could not close the owned Explorer window; runner disposal will close it.' } }
}

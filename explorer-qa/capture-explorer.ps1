# Windows PowerShell 5.1, STA. Disposable interactive Windows desktop only.
# Each view (in the ZIP, extracted) is captured on its own: the file's row where UI Automation gives its bounds, else only the folder
# view (the file list) after the folder, its content and the file are verified; otherwise nothing, with the reason.
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
    method = 'Windows Explorer Shell.Application + PrintWindow of that window, cut to the on-screen UI Automation row, else to its folder view (SHELLDLL_DefView) alone'
    app_executed = $false
}
$expectedNames = @()  # the public archive's own Ouroboros folder, read after its digest is checked

# The folder view of ONE Explorer window: a window class, not a localized name, and only that window's own descendants are walked.
$shellViewSource = @'
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Runtime.InteropServices;
using System.Text;
namespace ExplorerQa {
    public static class ShellView {
        [StructLayout(LayoutKind.Sequential)] private struct RECT { public int L, T, R, B; }
        private delegate bool EnumProc(IntPtr hwnd, IntPtr lParam);
        [DllImport("user32.dll")] private static extern bool EnumChildWindows(IntPtr parent, EnumProc callback, IntPtr lParam);
        [DllImport("user32.dll", CharSet = CharSet.Unicode)] private static extern int GetClassNameW(IntPtr hwnd, StringBuilder name, int max);
        [DllImport("user32.dll")] private static extern bool IsWindowVisible(IntPtr hwnd);
        [DllImport("user32.dll")] private static extern IntPtr GetAncestor(IntPtr hwnd, uint flags);
        [DllImport("user32.dll")] private static extern uint GetWindowThreadProcessId(IntPtr hwnd, out int pid);
        [DllImport("user32.dll")] private static extern bool GetWindowRect(IntPtr hwnd, out RECT r);

        // Screen rectangles of the visible SHELLDLL_DefView windows below "top", rooted at it (GA_ROOT) and owned by its process.
        public static Rectangle[] Visible(IntPtr top, int processId) {
            List<Rectangle> found = new List<Rectangle>();
            EnumProc callback = delegate (IntPtr hwnd, IntPtr lParam) {
                StringBuilder name = new StringBuilder(64);
                GetClassNameW(hwnd, name, name.Capacity);
                int owner;
                RECT r;
                if (name.ToString() == "SHELLDLL_DefView" && IsWindowVisible(hwnd) && GetAncestor(hwnd, 2) == top &&
                    GetWindowThreadProcessId(hwnd, out owner) != 0 && owner == processId && GetWindowRect(hwnd, out r)) {
                    found.Add(Rectangle.FromLTRB(r.L, r.T, r.R, r.B));
                }
                return true;
            };
            EnumChildWindows(top, callback, IntPtr.Zero);
            GC.KeepAlive(callback);
            return found.ToArray();
        }
    }
}
'@

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

# Why no row could be used, as counts only (no names): elements carrying the file's name, and how many items fail each demand.
function Get-RowCounts($bounds) {
    try {
        $root = [System.Windows.Automation.AutomationElement]::FromHandle($hwnd)
        $all = @($root.FindAll([System.Windows.Automation.TreeScope]::Descendants, $named))
        $items = @($all | Where-Object { $_.Current.ControlType.Id -in $ct::ListItem.Id, $ct::DataItem.Id })
        return [ordered]@{
            named = $all.Count; list_or_data_items = $items.Count
            offscreen = @($items | Where-Object { $_.Current.IsOffscreen }).Count
            empty_bounds = @($items | Where-Object { $_.Current.BoundingRectangle.IsEmpty }).Count
            other_process = @($items | Where-Object { $_.Current.ProcessId -ne $explorerProcess.Id }).Count
            beyond_window = @($items | Where-Object {
                $r = $_.Current.BoundingRectangle
                -not $r.IsEmpty -and ($r.Left -lt $bounds.Left -or $r.Right -gt $bounds.Right -or $r.Top -lt $bounds.Top -or $r.Bottom -gt $bounds.Bottom)
            }).Count
        }
    } catch { return [ordered]@{ unreadable = $_.Exception.GetType().Name } }
}

function Read-Frame([string]$label, [string]$folder) {
    $facts = [ordered]@{ status = 'not_captured'; frame_type = $null; stage = 'navigate'; skip_reason = 'folder_not_navigated'; error = $null
        shell = $null; window = $null; row = $null; row_unavailable = $null; content = $null; list_area = $null; png = $null }
    $report.frames[$label] = $facts
    $win.Navigate2($folder)
    $view = Wait-For {
        if (-not $win.Busy -and (Get-ViewPath $win.Document) -ieq $folder) { $win.Document }
    } "Explorer did not navigate to the required $label folder (ZIP namespace may be unavailable)."

    $facts.stage = 'shell_item'
    $facts.skip_reason = 'expected_file_not_exact'
    $item = $view.Folder.ParseName('Ouroboros.exe')
    if ($null -eq $item -or $item.IsFolder -or $item.Path -ine (Join-Path $folder 'Ouroboros.exe')) {
        throw "No exact Shell file in $label; refusing to capture a substitute row."
    }
    # Details = FVM_DETAILS (4). Re-read on the active view, after each navigation.
    $facts.skip_reason = 'details_view_not_accepted'
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
    $facts.skip_reason = 'window_not_usable'
    $bounds = [ExplorerQa.Native]::WindowBounds($hwnd)
    $dpi = [ExplorerQa.Native]::GetDpiForWindow($hwnd)
    $monitor = [Windows.Forms.Screen]::FromHandle($hwnd).Bounds
    if ($dpi -eq 0 -or $bounds.Width -lt 480 -or $bounds.Height -lt 340 -or -not $monitor.Contains($bounds)) {
        throw "Target must be unminimized, large enough and fully within one monitor in $label."
    }
    $facts.window = [ordered]@{ hwnd = $hwnd.ToInt64(); pid = $explorerProcess.Id; dpi = $dpi
        rect = @($bounds.X,$bounds.Y,$bounds.Width,$bounds.Height)
        monitor_rect = @($monitor.X,$monitor.Y,$monitor.Width,$monitor.Height) }
    if ($label -eq 'extracted' -and $report.frames['in-zip'].status -eq 'captured') {  # there is a pair to compare
        $first = $report.frames['in-zip'].window
        $facts.skip_reason = 'window_changed_since_the_first_view'
        if ($dpi -ne $first.dpi -or ($facts.window.rect -join ',') -ne ($first.rect -join ',') -or
            ($facts.window.monitor_rect -join ',') -ne ($first.monitor_rect -join ',')) {
            throw 'Window geometry, display or DPI changed between frames.'
        }
    }

    $row = $null
    try {
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
    } catch {
        Write-Warning $_.Exception.Message
        $facts.row_unavailable = [ordered]@{ reason = 'no_unique_row_with_readable_bounds_inside_the_window'; counts = (Get-RowCounts $bounds) }
    }
    if ($null -ne $row) {
        $current = $row.Current
        $r = $current.BoundingRectangle
        $rowPixels = [Drawing.Rectangle]::FromLTRB(
            [int][Math]::Ceiling($r.Left - $bounds.Left), [int][Math]::Ceiling($r.Top - $bounds.Top),
            [int][Math]::Floor($r.Right - $bounds.Left), [int][Math]::Floor($r.Bottom - $bounds.Top))
        $facts.row = [ordered]@{ name = $current.Name; control = $current.ControlType.ProgrammaticName
            rect_in_window = @($rowPixels.X,$rowPixels.Y,$rowPixels.Width,$rowPixels.Height) }
    } else {
        # The file list alone: only in the verified folder, holding nothing but the public archive's own entries, and its one folder view.
        $facts.stage = 'list_area'
        $facts.skip_reason = 'folder_changed_before_its_content_was_checked'
        if ((Get-ViewPath $win.Document) -ine $folder) { throw "Explorer left the $label folder." }
        $facts.skip_reason = 'folder_content_not_the_expected_synthetic_one'
        $shownNames = @(foreach ($entry in $win.Document.Folder.Items()) { [IO.Path]::GetFileName([string]$entry.Path) })
        $unexpected = @($shownNames | Where-Object { $_ -notin $expectedNames })
        $facts.content = [ordered]@{ items_shown = $shownNames.Count; items_in_the_archive_folder = $expectedNames.Count; unexpected = $unexpected.Count }
        if ($unexpected.Count -ne 0 -or 'Ouroboros.exe' -notin $shownNames) { throw "The $label view holds something other than the expected content." }
        $facts.content.brought_into_view = $true
        # SVSI_ENSUREVISIBLE (8) alone: scrolls to the file and selects nothing. If refused, the frame is judged by pixels.
        try { $win.Document.SelectItem($item, 8) } catch { $facts.content.brought_into_view = $false; Write-Warning $_.Exception.Message }
        Start-Sleep -Seconds 2  # shell icons are drawn after the rows
        $facts.skip_reason = 'folder_view_window_not_unique'
        $views = @([ExplorerQa.ShellView]::Visible($hwnd, $explorerProcess.Id))
        $facts.list_area = [ordered]@{ folder_view_windows = $views.Count; rect_in_window = $null }
        if ($views.Count -ne 1) { throw "$($views.Count) visible folder views in the $label window." }
        $area = $views[0]
        $facts.skip_reason = 'folder_view_not_a_part_of_the_window'
        if (-not $bounds.Contains($area) -or $area.Top -le $bounds.Top -or $area.Width -lt 240 -or $area.Height -lt 120) {
            throw "The folder view of $label is not a large enough part below the window's top edge."
        }
        $listPixels = [Drawing.Rectangle]::new($area.X - $bounds.X, $area.Y - $bounds.Y, $area.Width, $area.Height)
        $facts.list_area.rect_in_window = @($listPixels.X, $listPixels.Y, $listPixels.Width, $listPixels.Height)
    }
    $facts.stage = 'capture'
    # Re-bind immediately before pixels: tab changes and temporary overlays are not the test.
    $facts.skip_reason = 'view_changed_before_capture'
    if ((Get-ViewPath $win.Document) -ine $folder -or [int]$win.Document.CurrentViewMode -ne 4 -or
        [ExplorerQa.Native]::WindowBounds($hwnd) -ne $bounds -or
        [ExplorerQa.Native]::GetDpiForWindow($hwnd) -ne $dpi -or
        [int]$win.Document.SelectedItems().Count -ne 0) { throw "Folder/view/geometry changed or selection is nonempty in $label." }
    $facts.skip_reason = 'capture_failed_blank_or_changed'
    if ($null -ne $row) { $bitmap = [ExplorerQa.Native]::Capture($hwnd, $bounds, $rowPixels) }
    else { $bitmap = [ExplorerQa.Native]::Capture($hwnd, $bounds, $listPixels) }
    try {
        if ($win.Busy -or (Get-ViewPath $win.Document) -ine $folder -or
            -not [ExplorerQa.Native]::HasRowInk($bitmap, [Drawing.Rectangle]::new(0, 0, $bitmap.Width, $bitmap.Height))) { throw "Folder changed or the frame is blank in $label." }
        if ($null -eq $row) {
            $after = @([ExplorerQa.ShellView]::Visible($hwnd, $explorerProcess.Id))
            if ($after.Count -ne 1 -or $after[0] -ne $area) { throw "The folder view moved during the $label capture." }
        }
        $name = "$label.png"
        $png = Join-Path $evidence $name
        $bitmap.Save($png, [Drawing.Imaging.ImageFormat]::Png)
        $facts.png = [ordered]@{ file = $name; sha256 = (Get-FileHash -LiteralPath $png -Algorithm SHA256).Hash.ToLowerInvariant() }
    } finally { $bitmap.Dispose() }
    if ($null -ne $row) { $facts.frame_type = 'row' } else { $facts.frame_type = 'list_area' }
    $facts.stage = 'captured_pending_visual_inspection'
    $facts.status = 'captured'
    $facts.skip_reason = $null
}

$win = $null
$oldDpi = [IntPtr]::Zero
$incomplete = $true
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
    $archive = [IO.Compression.ZipFile]::OpenRead($zip)
    try {
        $expectedNames = @($archive.Entries | ForEach-Object { $_.FullName.Replace('\', '/') } |
            Where-Object { $_.StartsWith('Ouroboros/') -and $_.Length -gt 10 } |
            ForEach-Object { $_.Substring(10).Split('/')[0] } | Sort-Object -Unique)
    } finally { $archive.Dispose() }
    $report.source.folder_entries = $expectedNames.Count

    $report.stage = 'desktop'
    Add-Type -AssemblyName System.Drawing, System.Windows.Forms, UIAutomationClient, UIAutomationTypes, WindowsBase
    Add-Type -Path (Join-Path $PSScriptRoot 'ExplorerCapture.cs') -ReferencedAssemblies System.Drawing
    Add-Type -TypeDefinition $shellViewSource -ReferencedAssemblies System.Drawing
    $oldDpi = [ExplorerQa.Native]::SetThreadDpiAwarenessContext([IntPtr](-4)) # Per-monitor V2.
    if ($oldDpi -eq [IntPtr]::Zero) { throw 'Failed to bind capture thread to physical pixels.' }
    $ae = [System.Windows.Automation.AutomationElement]
    $ct = [System.Windows.Automation.ControlType]
    $named = [System.Windows.Automation.OrCondition]::new(
        [System.Windows.Automation.PropertyCondition]::new($ae::NameProperty, 'Ouroboros.exe'),
        [System.Windows.Automation.PropertyCondition]::new($ae::NameProperty, 'Ouroboros'))
    $namedItem = [System.Windows.Automation.AndCondition]::new($named,
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
    foreach ($target in @(@('in-zip', ($zip + '\Ouroboros')), @('extracted', $diskFolder))) {
        $report.stage = $target[0]
        try { Read-Frame $target[0] $target[1] }  # a view that fails never stops the other one
        catch {
            $report.frames[$target[0]].error = $_.Exception.GetType().Name  # the message stays in the job log
            Write-Warning "$($target[0]) not captured: $($_.Exception.Message)"
        }
    }
    $captured = @($report.frames.Keys | Where-Object { $report.frames[$_].status -eq 'captured' })
    $report.captured_views = $captured
    $report.stage = 'capture_complete'
    $report.visual = if ($captured.Count -eq 2) { 'pending_visual_inspection' } elseif ($captured.Count -eq 1) { 'partial_pending_visual_inspection' } else { 'failed_to_capture' }
    $incomplete = $captured.Count -ne 2
    Write-Host "Explorer views captured: $($captured.Count) of 2; icon appearance still requires visual inspection."
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
if ($incomplete) { exit 1 }  # a partial result is reported and uploaded, never a success

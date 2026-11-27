# Windows PowerShell 5.1. Parses the kit's own scripts (this folder and the kit root) with PowerShell's own parser.
# Nothing is executed. Run first on the disposable host so a typo costs seconds, not an installed-and-launched run.
[CmdletBinding()]
param([Parameter(Mandatory)][string]$OutJson)
Set-StrictMode -Version 3
$ErrorActionPreference = 'Stop'
$errors = @()
$names = @()
$folders = @($PSScriptRoot, (Split-Path -Parent $PSScriptRoot))
foreach ($file in @($folders | ForEach-Object { Get-ChildItem -LiteralPath $_ -Filter '*.ps1' -File })) {
    $names += $file.Name
    $tokens = $null
    $parseErrors = $null
    [void][System.Management.Automation.Language.Parser]::ParseFile($file.FullName, [ref]$tokens, [ref]$parseErrors)
    if ($parseErrors) {
        foreach ($e in $parseErrors) { $errors += ($file.Name + ':' + $e.Extent.StartLineNumber + ': ' + $e.ErrorId) }
    }
}
$out = [ordered]@{ ok = ($errors.Count -eq 0 -and $names.Count -ge 4); files = $names; errors = $errors }
$out | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $OutJson -Encoding UTF8

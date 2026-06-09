param(
    [string]$OutputPath = "dist/autodl_project.zip"
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$DistDir = Join-Path $RepoRoot "dist"
$TempDir = Join-Path $DistDir "autodl-package"

function Resolve-RepoPath {
    param([string]$Path)
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return $Path
    }
    return Join-Path $RepoRoot $Path
}

function Remove-TempDir {
    if (-not (Test-Path -LiteralPath $TempDir)) {
        return
    }
    $resolvedDist = (Resolve-Path -LiteralPath $DistDir).Path
    $resolvedTemp = (Resolve-Path -LiteralPath $TempDir).Path
    if (-not $resolvedTemp.StartsWith($resolvedDist, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove temp directory outside dist: $resolvedTemp"
    }
    Remove-Item -LiteralPath $TempDir -Recurse -Force
}

New-Item -ItemType Directory -Force -Path $DistDir | Out-Null
Remove-TempDir
New-Item -ItemType Directory -Force -Path $TempDir | Out-Null

$files = @()
try {
    Push-Location $RepoRoot
    $files = git ls-files --cached --others --exclude-standard
    if ($LASTEXITCODE -ne 0) {
        $files = @()
    }
}
finally {
    Pop-Location
}

if ($files.Count -eq 0) {
    $excludedDirs = @(".git", ".pytest_cache", ".pytest-tmp", ".venv", "venv", "data", "checkpoints", "outputs", "dist")
    $files = Get-ChildItem -Path $RepoRoot -Recurse -File |
        Where-Object {
            $relative = [System.IO.Path]::GetRelativePath($RepoRoot, $_.FullName)
            $parts = $relative -split '[\\/]'
            -not ($parts | Where-Object { $excludedDirs -contains $_ })
        } |
        ForEach-Object { [System.IO.Path]::GetRelativePath($RepoRoot, $_.FullName) }
}

foreach ($file in $files) {
    $source = Join-Path $RepoRoot $file
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        continue
    }
    $destination = Join-Path $TempDir $file
    $destinationDir = Split-Path -Parent $destination
    New-Item -ItemType Directory -Force -Path $destinationDir | Out-Null
    Copy-Item -LiteralPath $source -Destination $destination -Force
}

$resolvedOutput = Resolve-RepoPath $OutputPath
$resolvedOutputDir = Split-Path -Parent $resolvedOutput
if ($resolvedOutputDir) {
    New-Item -ItemType Directory -Force -Path $resolvedOutputDir | Out-Null
}

if (Test-Path -LiteralPath $resolvedOutput) {
    Remove-Item -LiteralPath $resolvedOutput -Force
}

Compress-Archive -Path (Join-Path $TempDir "*") -DestinationPath $resolvedOutput -Force
Remove-TempDir

Write-Host "Created $resolvedOutput"

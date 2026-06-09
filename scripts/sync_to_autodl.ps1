param(
    [Parameter(Mandatory = $true)]
    [string]$HostName,

    [string]$User = "root",
    [int]$Port = 22,
    [string]$RemoteDir = "~/int4-ptq",
    [string]$ArchivePath = "dist/autodl_project.zip",
    [switch]$NoUnpack
)

$ErrorActionPreference = "Stop"

if ($RemoteDir -match "\s") {
    throw "RemoteDir must not contain spaces. Use a path like ~/int4-ptq."
}

foreach ($commandName in @("ssh", "scp")) {
    if (-not (Get-Command $commandName -ErrorAction SilentlyContinue)) {
        throw "Missing required command: $commandName"
    }
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PackageScript = Join-Path $PSScriptRoot "package_for_autodl.ps1"

& $PackageScript -OutputPath $ArchivePath
if ($LASTEXITCODE -ne 0) {
    throw "Packaging failed."
}

if ([System.IO.Path]::IsPathRooted($ArchivePath)) {
    $resolvedArchive = (Resolve-Path -LiteralPath $ArchivePath).Path
}
else {
    $resolvedArchive = (Resolve-Path -LiteralPath (Join-Path $RepoRoot $ArchivePath)).Path
}

$target = if ($User.Trim()) { "$User@$HostName" } else { $HostName }
$remoteArchive = "$RemoteDir/autodl_project.zip"

ssh -p $Port $target "mkdir -p $RemoteDir"
if ($LASTEXITCODE -ne 0) {
    throw "Remote directory creation failed."
}

scp -P $Port $resolvedArchive "${target}:$remoteArchive"
if ($LASTEXITCODE -ne 0) {
    throw "Upload failed."
}

if (-not $NoUnpack) {
    ssh -p $Port $target "cd $RemoteDir && unzip -o autodl_project.zip >/dev/null && chmod +x scripts/*.sh"
    if ($LASTEXITCODE -ne 0) {
        throw "Remote unpack failed."
    }
}

Write-Host "Synced to ${target}:$RemoteDir"

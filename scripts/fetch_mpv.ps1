$ErrorActionPreference = "Stop"

$tag = "2026-09-20-e76a35ec95"
$asset = "mpv-dev-lgpl-x86_64-20260920-git-e76a35ec95.7z"
$expected = "c8d52781ed8773bf414faf12aa74415ecdf4f2ecc7254a5d633d76266d131b9d"
$url = "https://github.com/zhongfly/mpv-winbuild/releases/download/$tag/$asset"

$outDir = Join-Path $PSScriptRoot "..\mpv-runtime"
$archive = Join-Path $PSScriptRoot "..\$asset"

if (Test-Path $outDir) {
    Remove-Item $outDir -Recurse -Force
}
New-Item -ItemType Directory -Path $outDir | Out-Null

Write-Host "Downloading LGPL-oriented libmpv runtime..."
Invoke-WebRequest -Uri $url -OutFile $archive

$actual = (Get-FileHash -Path $archive -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actual -ne $expected) {
    throw "libmpv SHA256 mismatch. Expected $expected, got $actual"
}

$sevenZip = Get-Command 7z -ErrorAction SilentlyContinue
if (-not $sevenZip) {
    throw "7z is required to extract the pinned libmpv runtime."
}

& $sevenZip.Source x $archive "-o$outDir" -y | Out-Host
if ($LASTEXITCODE -ne 0) {
    throw "Failed to extract libmpv runtime."
}

$dll = Get-ChildItem $outDir -Recurse -File | Where-Object {
    $_.Name -match "^(libmpv|mpv)-?2?\.dll$"
} | Select-Object -First 1

if (-not $dll) {
    throw "libmpv DLL not found in extracted runtime."
}

Write-Host "libmpv runtime ready: $($dll.FullName)"

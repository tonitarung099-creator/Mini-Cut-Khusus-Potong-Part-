$ErrorActionPreference = "Stop"

$tag = "autobuild-2026-08-31-13-27"
$asset = "ffmpeg-N-126342-gf88b741dbf-win64-lgpl.zip"
$expected = "7a7d7ad65d5d53aefc57fc3f78e00febe7b65156c03d14d43efc1393ef46a111"
$url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/$tag/$asset"

$root = Join-Path $PSScriptRoot ".."
$outDir = Join-Path $root "ffmpeg-runtime"
$archive = Join-Path $root $asset

if (Test-Path $outDir) {
    Remove-Item $outDir -Recurse -Force
}
New-Item -ItemType Directory -Path $outDir | Out-Null

Write-Host "Downloading pinned static LGPL FFmpeg runtime..."
Invoke-WebRequest -Uri $url -OutFile $archive

$actual = (Get-FileHash -Path $archive -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actual -ne $expected) {
    throw "FFmpeg SHA256 mismatch. Expected $expected, got $actual"
}

$extractDir = Join-Path $outDir "_extract"
Expand-Archive -Path $archive -DestinationPath $extractDir -Force

$ffmpeg = Get-ChildItem $extractDir -Recurse -File -Filter "ffmpeg.exe" | Select-Object -First 1
$ffprobe = Get-ChildItem $extractDir -Recurse -File -Filter "ffprobe.exe" | Select-Object -First 1
if (-not $ffmpeg -or -not $ffprobe) {
    throw "ffmpeg.exe/ffprobe.exe not found in pinned archive."
}

Copy-Item $ffmpeg.FullName (Join-Path $outDir "ffmpeg.exe") -Force
Copy-Item $ffprobe.FullName (Join-Path $outDir "ffprobe.exe") -Force

$license = Get-ChildItem $extractDir -Recurse -File | Where-Object {
    $_.Name -match "^(LICENSE|COPYING)(\..*)?$"
} | Select-Object -First 1
if ($license) {
    Copy-Item $license.FullName (Join-Path $outDir "FFMPEG_BUILD_LICENSE.txt") -Force
}

Remove-Item $extractDir -Recurse -Force
Remove-Item $archive -Force

& (Join-Path $outDir "ffmpeg.exe") -version | Select-Object -First 1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& (Join-Path $outDir "ffprobe.exe") -version | Select-Object -First 1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "FFmpeg runtime ready: $outDir"

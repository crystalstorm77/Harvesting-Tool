# ============================================================
# SECTION A - Parameters And Defaults
# ============================================================

param(
    [string]$VideoPath = "A:\Desktop\Crap to Sort\Automation Projects\Harvesting Tool\Test footage\HB-Test-Venonat.mp4",
    [string]$ChapterStart = "00:00:12:06",
    [string]$ChapterEnd = "00:10:08:00",
    [string]$MinHarvest = "00:01:00:00",
    [string]$MaxHarvest = "00:03:00:00",
    [string]$MinClipLength = "00:00:00:15",
    [string]$MaxClipLength = "00:00:07:00",
    [string]$PauseThreshold = "00:00:05:00",
    [int]$LeadInSeconds = 0,
    [int]$LeadInFrames = 2,
    [int]$TailAfterSeconds = 0,
    [int]$TailAfterFrames = 4,
    [string]$OutputStem = "A:\Desktop\Crap to Sort\Automation Projects\Harvesting Tool\Test footage\hb_test_smoke",
    [string]$DebugStem = "A:\Desktop\Crap to Sort\Automation Projects\Harvesting Tool\Test footage\hb_test_smoke_debug",
    [int]$SampleStride = 2,
    [double]$ActivityThreshold = 8.0,
    [double]$ActivePixelRatio = 0.0015,
    [string]$MinBurst = "00:00:00:06"
)


# ============================================================
# SECTION B - Repo Setup
# ============================================================

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptRoot
$env:PYTHONPATH = "src"

Write-Host "Running smoke test from $scriptRoot"
Write-Host "Video: $VideoPath"
Write-Host "Chapter: $ChapterStart -> $ChapterEnd"
Write-Host "Output: $OutputStem.txt and $OutputStem.json"
Write-Host "Debug: $DebugStem"
Write-Host ""


# ============================================================
# SECTION C - Smoke Test Execution
# ============================================================

py -m harvesting_tool.cli $VideoPath `
    --chapter-start $ChapterStart `
    --chapter-end $ChapterEnd `
    --min-harvest $MinHarvest `
    --max-harvest $MaxHarvest `
    --min-clip-length $MinClipLength `
    --max-clip-length $MaxClipLength `
    --pause-threshold $PauseThreshold `
    --lead-in-seconds $LeadInSeconds `
    --lead-in-frames $LeadInFrames `
    --tail-after-seconds $TailAfterSeconds `
    --tail-after-frames $TailAfterFrames `
    --output-stem $OutputStem `
    --debug-stem $DebugStem `
    --sample-stride $SampleStride `
    --activity-threshold $ActivityThreshold `
    --active-pixel-ratio $ActivePixelRatio `
    --min-burst $MinBurst

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Smoke test finished successfully."
} else {
    Write-Host ""
    Write-Host "Smoke test failed with exit code $LASTEXITCODE."
    exit $LASTEXITCODE
}

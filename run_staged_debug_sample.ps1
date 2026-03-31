# ============================================================
# SECTION A - Parameters And Defaults
# ============================================================

param(
    [string]$VideoPath = "A:\Desktop\Crap to Sort\Automation Projects\Harvesting Tool\Test footage\Mini-Sample-2\Mini-Sample-2.mp4",
    [string]$ChapterStart = "00:00:00:00",
    [string]$ChapterEnd = "00:00:58:01",
    [string]$MinHarvest = "00:00:05:00",
    [string]$MaxHarvest = "00:02:00:00",
    [string]$MinClipLength = "00:00:00:15",
    [string]$MaxClipLength = "00:00:30:00",
    [string]$PauseThreshold = "00:00:05:00",
    [int]$LeadInSeconds = 0,
    [int]$LeadInFrames = 2,
    [int]$TailAfterSeconds = 0,
    [int]$TailAfterFrames = 4,
    [string]$OutputStem = "A:\Desktop\Crap to Sort\Automation Projects\Harvesting Tool\Test footage\mini_sample2_staged_debug_longclips",
    [string]$DebugStem = "A:\Desktop\Crap to Sort\Automation Projects\Harvesting Tool\Test footage\mini_sample2_staged_debug_longclips_debug",
    [int]$SampleStride = 2,
    [double]$ActivityThreshold = 8.0,
    [double]$ActivePixelRatio = 0.0015,
    [string]$MinBurst = "00:00:00:06",
    [switch]$OpenOutputFolder
)


# ============================================================
# SECTION B - Repo Setup And Editable Guidance
# ============================================================

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptRoot
$env:PYTHONPATH = "src"

Write-Host "Running staged debug harvest from $scriptRoot"
Write-Host "Video: $VideoPath"
Write-Host "Chapter: $ChapterStart -> $ChapterEnd"
Write-Host "Cut list stem: $OutputStem"
Write-Host "Debug stem: $DebugStem"
Write-Host ""
Write-Host "Edit these values when you want to target a different sample:"
Write-Host "  -VideoPath"
Write-Host "  -ChapterStart / -ChapterEnd"
Write-Host "  -OutputStem / -DebugStem"
Write-Host ""


# ============================================================
# SECTION C - Staged Detector Execution
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
    --min-burst $MinBurst `
    --use-staged-detector

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Staged debug harvest failed with exit code $LASTEXITCODE."
    exit $LASTEXITCODE
}


# ============================================================
# SECTION D - Result Summary And Resolve Hand-Off
# ============================================================

$outputJsonPath = "$OutputStem.json"
$outputTextPath = "$OutputStem.txt"
$debugDirectory = Split-Path -Parent $DebugStem
$resolveScriptPath = Join-Path $scriptRoot "resolve_scripts\assemble_review_timeline_from_json.py"

Write-Host ""
Write-Host "Staged debug harvest finished successfully."
Write-Host "Cut list text: $outputTextPath"
Write-Host "Cut list json: $outputJsonPath"
Write-Host "Debug artifacts prefix: $DebugStem"
Write-Host ""
Write-Host "DaVinci Resolve Console command:"
Write-Host "exec(open(r'$resolveScriptPath', encoding='utf-8').read(), globals())"
Write-Host ""
Write-Host "Before you run it, edit the settings at the top of:"
Write-Host "  $resolveScriptPath"
Write-Host "Set HARVEST_JSON_PATH to:"
Write-Host "  $outputJsonPath"
Write-Host ""

if ($OpenOutputFolder) {
    Start-Process explorer.exe $debugDirectory
}

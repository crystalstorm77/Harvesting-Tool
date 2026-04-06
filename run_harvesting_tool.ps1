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
    [string]$OutputStem = "A:\Desktop\Crap to Sort\Automation Projects\Harvesting Tool\Test footage\hb_test_harvest",
    [string]$DebugStem = "A:\Desktop\Crap to Sort\Automation Projects\Harvesting Tool\Test footage\hb_test_harvest_debug",
    [int]$SampleStride = 1,
    [double]$ActivityThreshold = 8.0,
    [double]$ActivePixelRatio = 0.0015,
    [string]$MinBurst = "00:00:00:06",
    [string]$ResolveTimelineName = "Harvest Review",
    [switch]$UseStagedDetector,
    [switch]$StagedStage3ArtStatePrototype
)


# ============================================================
# SECTION B - Repo Setup And Resolve Hand-Off
# ============================================================

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptRoot
$env:PYTHONPATH = "src"
$resolveScriptPath = Join-Path $scriptRoot 'resolve_scripts\assemble_review_timeline_from_json.py'
$outputTextPath = Join-Path (Split-Path -Parent $OutputStem) ((Split-Path -Leaf $OutputStem) + ' - Final Harvested Clips.txt')
$outputJsonPath = Join-Path (Split-Path -Parent $OutputStem) ((Split-Path -Leaf $OutputStem) + ' - Final Harvested Clips - Machine Readable.json')

Write-Host "Running harvesting tool from $scriptRoot"
Write-Host "Video: $VideoPath"
Write-Host "Chapter: $ChapterStart -> $ChapterEnd"
Write-Host "Output: $outputTextPath and $outputJsonPath"
Write-Host "Debug: $DebugStem"
Write-Host "Resolve timeline name: $ResolveTimelineName"
Write-Host "Use staged detector: $UseStagedDetector"
Write-Host ""
# SECTION C - Harvesting Tool Execution
# ============================================================

$cliArgs = @(
    '-m', 'harvesting_tool.cli',
    $VideoPath,
    '--chapter-start', $ChapterStart,
    '--chapter-end', $ChapterEnd,
    '--min-harvest', $MinHarvest,
    '--max-harvest', $MaxHarvest,
    '--min-clip-length', $MinClipLength,
    '--max-clip-length', $MaxClipLength,
    '--pause-threshold', $PauseThreshold,
    '--lead-in-seconds', $LeadInSeconds,
    '--lead-in-frames', $LeadInFrames,
    '--tail-after-seconds', $TailAfterSeconds,
    '--tail-after-frames', $TailAfterFrames,
    '--output-stem', $OutputStem,
    '--debug-stem', $DebugStem,
    '--sample-stride', $SampleStride,
    '--activity-threshold', $ActivityThreshold,
    '--active-pixel-ratio', $ActivePixelRatio,
    '--min-burst', $MinBurst
)

if ($UseStagedDetector) {
    $cliArgs += '--use-staged-detector'
}
if ($StagedStage3ArtStatePrototype) {
    $cliArgs += '--staged-stage3-art-state-prototype'
}

py @cliArgs
$exitCode = $LASTEXITCODE

Write-Host ""
if ($exitCode -eq 0) {
    $escapedResolveScriptPath = $resolveScriptPath.Replace("'", "''")
    $escapedOutputJsonPath = $outputJsonPath.Replace("'", "''")
    $escapedTimelineName = $ResolveTimelineName.Replace("'", "''")
    $resolveCommand = "import os; os.environ['HARVEST_JSON_PATH'] = r'$escapedOutputJsonPath'; os.environ['REVIEW_TIMELINE_NAME'] = r'$escapedTimelineName'; exec(open(r'$escapedResolveScriptPath', encoding='utf-8').read(), globals())"

    Write-Host 'Harvesting tool run finished successfully.' -ForegroundColor Green
    Write-Host "Cut list text: $outputTextPath"
    Write-Host "Cut list json: $outputJsonPath"
    Write-Host "Debug artifacts prefix: $DebugStem"
    Write-Host ''
    Write-Host 'DaVinci Resolve Console command:' -ForegroundColor Cyan
    Write-Host $resolveCommand
} else {
    Write-Host "Harvesting tool run failed with exit code $exitCode." -ForegroundColor Red
    exit $exitCode
}

# ============================================================
# SECTION A - Interactive Prompt Helpers
# ============================================================

function Read-HostWithDefault {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Prompt,
        [Parameter(Mandatory = $true)]
        [string]$DefaultValue
    )

    $response = Read-Host "$Prompt [$DefaultValue]"
    if ([string]::IsNullOrWhiteSpace($response)) {
        return $DefaultValue
    }

    return $response.Trim()
}

function Read-YesNoWithDefault {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Prompt,
        [Parameter(Mandatory = $true)]
        [bool]$DefaultValue
    )

    $defaultLabel = if ($DefaultValue) { 'Y' } else { 'N' }

    while ($true) {
        $response = Read-Host "$Prompt [Y/N, default: $defaultLabel]"
        if ([string]::IsNullOrWhiteSpace($response)) {
            return $DefaultValue
        }

        switch ($response.Trim().ToLowerInvariant()) {
            'y' { return $true }
            'yes' { return $true }
            'n' { return $false }
            'no' { return $false }
            default {
                Write-Host 'Please enter Y or N.' -ForegroundColor Yellow
            }
        }
    }
}

function Get-SafeName {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RawName,
        [Parameter(Mandatory = $true)]
        [string]$Fallback
    )

    $invalidChars = [System.IO.Path]::GetInvalidFileNameChars()
    $safeChars = $RawName.ToCharArray() | ForEach-Object {
        if ($invalidChars -contains $_) {
            '_'
        }
        elseif ([char]::IsWhiteSpace($_)) {
            '_'
        }
        else {
            $_
        }
    }

    $safeName = (-join $safeChars).Trim('_')
    if ([string]::IsNullOrWhiteSpace($safeName)) {
        return $Fallback
    }

    return $safeName
}


# ============================================================
# SECTION B - Interactive Prompt Flow
# ============================================================

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$defaultOutputRoot = Join-Path $scriptRoot 'Test footage'
$defaultVideoPath = Join-Path $defaultOutputRoot 'Sample 2\Sample 2.mp4'
$defaultChapterStart = '00:00:00:00'
$defaultChapterEnd = '00:03:38:29'
$defaultMinHarvest = '00:00:10:00'
$defaultMaxHarvest = '00:05:00:00'
$defaultMinClipLength = '00:00:00:15'
$defaultMaxClipLength = '00:10:00:00'
$defaultPauseThreshold = '00:00:05:00'
$defaultLeadInSeconds = '0'
$defaultLeadInFrames = '2'
$defaultTailAfterSeconds = '0'
$defaultTailAfterFrames = '4'
$defaultSampleStride = '1'
$defaultActivityThreshold = '8.0'
$defaultActivePixelRatio = '0.0015'
$defaultMinBurst = '00:00:00:10'
$defaultUseLegacyDetector = $false

Write-Host ''
Write-Host 'Interactive harvesting tool runner' -ForegroundColor Cyan
Write-Host 'Leave any prompt blank to accept the default shown in brackets.'
Write-Host ''

$videoPath = Read-HostWithDefault -Prompt 'Video path' -DefaultValue $defaultVideoPath
$precomputedMovementEvidenceJson = Read-Host 'Pre-made Stage 1A Movement Evidence Record.json path (leave blank to scan video)'
if (-not [string]::IsNullOrWhiteSpace($precomputedMovementEvidenceJson)) {
    $precomputedMovementEvidenceJson = $precomputedMovementEvidenceJson.Trim()
}
$videoBaseName = [System.IO.Path]::GetFileNameWithoutExtension($videoPath)
$defaultStemName = if ([string]::IsNullOrWhiteSpace($videoBaseName)) { 'harvest_run' } else { "$videoBaseName`_harvest" }
$stemNameInput = Read-HostWithDefault -Prompt 'Output name stem (file-name only, no extension)' -DefaultValue $defaultStemName
$safeStemName = Get-SafeName -RawName $stemNameInput -Fallback 'harvest_run'
$defaultFolderName = "$safeStemName $((Get-Date).ToString('yyyy-MM-dd-HH-mm-ss'))"
$folderNameInput = Read-HostWithDefault -Prompt 'Output folder name inside Test footage' -DefaultValue $defaultFolderName
$safeFolderName = Get-SafeName -RawName $folderNameInput -Fallback $defaultFolderName
$runOutputRoot = Join-Path $defaultOutputRoot $safeFolderName
$defaultResolveTimelineName = $safeStemName
$resolveTimelineName = Read-HostWithDefault -Prompt 'DaVinci Resolve review timeline name' -DefaultValue $defaultResolveTimelineName

$chapterStart = Read-HostWithDefault -Prompt 'Chapter start timecode' -DefaultValue $defaultChapterStart
$chapterEnd = Read-HostWithDefault -Prompt 'Chapter end timecode' -DefaultValue $defaultChapterEnd
$minHarvest = Read-HostWithDefault -Prompt 'Minimum total harvest time' -DefaultValue $defaultMinHarvest
$maxHarvest = Read-HostWithDefault -Prompt 'Maximum total harvest time' -DefaultValue $defaultMaxHarvest
$minClipLength = Read-HostWithDefault -Prompt 'Minimum clip length' -DefaultValue $defaultMinClipLength
$maxClipLength = Read-HostWithDefault -Prompt 'Maximum clip length' -DefaultValue $defaultMaxClipLength
$pauseThreshold = Read-HostWithDefault -Prompt 'Pause threshold' -DefaultValue $defaultPauseThreshold
$leadInSeconds = Read-HostWithDefault -Prompt 'Lead-in seconds' -DefaultValue $defaultLeadInSeconds
$leadInFrames = Read-HostWithDefault -Prompt 'Lead-in frames' -DefaultValue $defaultLeadInFrames
$tailAfterSeconds = Read-HostWithDefault -Prompt 'Tail-after seconds' -DefaultValue $defaultTailAfterSeconds
$tailAfterFrames = Read-HostWithDefault -Prompt 'Tail-after frames' -DefaultValue $defaultTailAfterFrames
$sampleStride = Read-HostWithDefault -Prompt 'Sample stride' -DefaultValue $defaultSampleStride
$activityThreshold = Read-HostWithDefault -Prompt 'Activity threshold' -DefaultValue $defaultActivityThreshold
$activePixelRatio = Read-HostWithDefault -Prompt 'Active pixel ratio' -DefaultValue $defaultActivePixelRatio
$minBurst = Read-HostWithDefault -Prompt 'Minimum burst duration' -DefaultValue $defaultMinBurst
$useLegacyDetector = Read-YesNoWithDefault -Prompt 'Use legacy detector instead of V3 staged detector' -DefaultValue $defaultUseLegacyDetector

$outputStem = Join-Path $runOutputRoot $safeStemName
$debugStem = Join-Path $runOutputRoot ($safeStemName + '_debug')

Write-Host ''
Write-Host 'About to run with these settings:' -ForegroundColor Cyan
Write-Host "  Video: $videoPath"
Write-Host "  Output folder: $runOutputRoot"
Write-Host "  Chapter: $chapterStart -> $chapterEnd"
Write-Host "  Output stem: $outputStem"
Write-Host "  Debug stem: $debugStem"
Write-Host "  Resolve timeline name: $resolveTimelineName"
Write-Host "  Use legacy detector: $useLegacyDetector"
if (-not [string]::IsNullOrWhiteSpace($precomputedMovementEvidenceJson)) {
    Write-Host "  Pre-made Stage 1A record: $precomputedMovementEvidenceJson"
}
Write-Host ''

$shouldRun = Read-YesNoWithDefault -Prompt 'Start the harvesting tool now' -DefaultValue $true
if (-not $shouldRun) {
    Write-Host ''
    Write-Host 'Cancelled before running.' -ForegroundColor Yellow
    Read-Host 'Press Enter to close'
    exit 0
}


# ============================================================
# SECTION C - Harvesting Tool Invocation
# ============================================================

$engineScriptPath = Join-Path $scriptRoot 'run_harvesting_tool.ps1'

if (-not (Test-Path -LiteralPath $engineScriptPath)) {
    Write-Host ''
    Write-Host "Could not find harvesting tool engine at: $engineScriptPath" -ForegroundColor Red
    Read-Host 'Press Enter to close'
    exit 1
}

if (-not (Test-Path -LiteralPath $defaultOutputRoot)) {
    New-Item -ItemType Directory -Path $defaultOutputRoot | Out-Null
}

if (-not (Test-Path -LiteralPath $runOutputRoot)) {
    New-Item -ItemType Directory -Path $runOutputRoot | Out-Null
}

$engineArgs = @(
    '-ExecutionPolicy', 'Bypass',
    '-File', $engineScriptPath,
    '-VideoPath', $videoPath,
    '-ChapterStart', $chapterStart,
    '-ChapterEnd', $chapterEnd,
    '-MinHarvest', $minHarvest,
    '-MaxHarvest', $maxHarvest,
    '-MinClipLength', $minClipLength,
    '-MaxClipLength', $maxClipLength,
    '-PauseThreshold', $pauseThreshold,
    '-LeadInSeconds', ([int]$leadInSeconds).ToString(),
    '-LeadInFrames', ([int]$leadInFrames).ToString(),
    '-TailAfterSeconds', ([int]$tailAfterSeconds).ToString(),
    '-TailAfterFrames', ([int]$tailAfterFrames).ToString(),
    '-OutputStem', $outputStem,
    '-DebugStem', $debugStem,
    '-SampleStride', ([int]$sampleStride).ToString(),
    '-ActivityThreshold', ([double]$activityThreshold).ToString([System.Globalization.CultureInfo]::InvariantCulture),
    '-ActivePixelRatio', ([double]$activePixelRatio).ToString([System.Globalization.CultureInfo]::InvariantCulture),
    '-MinBurst', $minBurst,
    '-ResolveTimelineName', $resolveTimelineName
)

if (-not [string]::IsNullOrWhiteSpace($precomputedMovementEvidenceJson)) {
    $engineArgs += '-PrecomputedMovementEvidenceJson'
    $engineArgs += $precomputedMovementEvidenceJson
}
if ($useLegacyDetector) {
    $engineArgs += '-UseLegacyDetector'
}

Write-Host ''
Write-Host 'Launching harvesting tool...' -ForegroundColor Cyan
Write-Host ''

& powershell @engineArgs
$exitCode = $LASTEXITCODE

Write-Host ''
if ($exitCode -eq 0) {
    Write-Host 'Interactive harvesting tool run finished successfully.' -ForegroundColor Green
    Write-Host "Results should now be under: $runOutputRoot"
}
else {
    Write-Host "Interactive harvesting tool run failed with exit code $exitCode." -ForegroundColor Red
}

Read-Host 'Press Enter to close'
exit $exitCode



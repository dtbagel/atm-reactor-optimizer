$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$outputDir = Join-Path $taskRoot 'outputs'
$buildDir = Join-Path $taskRoot 'build'
New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
foreach ($fileName in @('atm_er2_optimizer.exe','nvrtc64_120_0.dll','nvrtc-builtins64_129.dll')) {
    Copy-Item -LiteralPath (Join-Path $buildDir $fileName) -Destination $outputDir -Force
}
Copy-Item -LiteralPath (Join-Path $taskRoot 'README.md') -Destination $outputDir -Force
$sourceStage = Join-Path $taskRoot 'work\package_source'
$stageTools = Join-Path $sourceStage 'tools'
New-Item -ItemType Directory -Path $stageTools -Force | Out-Null
foreach ($fileName in @('CMakeLists.txt','CMakePresets.json','README.md','ato.py')) {
    Copy-Item -LiteralPath (Join-Path $taskRoot $fileName) -Destination $sourceStage -Force
}
foreach ($directoryName in @('src','tests')) {
    Copy-Item -LiteralPath (Join-Path $taskRoot $directoryName) -Destination $sourceStage -Recurse -Force
}
foreach ($fileName in @('bootstrap.py','build.ps1','profile.ps1','package.ps1')) {
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot $fileName) -Destination $stageTools -Force
}
Compress-Archive -Path (Join-Path $sourceStage '*') -DestinationPath (Join-Path $outputDir 'cpp_source.zip') -Force
Write-Output "Packaged executable, NVIDIA runtime, documentation, and source in $outputDir"

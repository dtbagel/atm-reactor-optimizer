param([string]$Destination)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$outputDir = if ($Destination) { [IO.Path]::GetFullPath($Destination) } else { Join-Path $taskRoot 'outputs' }
$buildDir = Join-Path $taskRoot 'build'
New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
foreach ($fileName in @('atm_er2_gui.exe','atm_er2_optimizer.exe','nvrtc64_120_0.dll','nvrtc-builtins64_129.dll')) {
    Copy-Item -LiteralPath (Join-Path $buildDir $fileName) -Destination $outputDir -Force
}
Copy-Item -LiteralPath (Join-Path $taskRoot 'README.md') -Destination $outputDir -Force
Copy-Item -LiteralPath (Join-Path $taskRoot 'OPTI.md') -Destination $outputDir -Force
Copy-Item -LiteralPath (Join-Path $taskRoot 'PERFORMANCE.md') -Destination $outputDir -Force
Copy-Item -LiteralPath (Join-Path $taskRoot 'LICENSE') -Destination $outputDir -Force
Copy-Item -LiteralPath (Join-Path $taskRoot 'vendor\imgui\LICENSE.txt') -Destination (Join-Path $outputDir 'Dear-ImGui-LICENSE.txt') -Force
$sourceStage = Join-Path $taskRoot ('work\package_source_' + [guid]::NewGuid().ToString('N'))
$stageTools = Join-Path $sourceStage 'tools'
New-Item -ItemType Directory -Path $stageTools -Force | Out-Null
foreach ($fileName in @('CMakeLists.txt','CMakePresets.json','README.md','ato.py','LICENSE','CONTRIBUTORS.md','OPTI.md','PERFORMANCE.md')) {
    Copy-Item -LiteralPath (Join-Path $taskRoot $fileName) -Destination $sourceStage -Force
}
foreach ($directoryName in @('src','tests','data','vendor')) {
    Copy-Item -LiteralPath (Join-Path $taskRoot $directoryName) -Destination $sourceStage -Recurse -Force
}
foreach ($fileName in @('bootstrap.py','build.ps1','profile.ps1','package.ps1','generate_presets.py','performance.cpp')) {
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot $fileName) -Destination $stageTools -Force
}
Compress-Archive -Path (Join-Path $sourceStage '*') -DestinationPath (Join-Path $outputDir 'cpp_source.zip') -Force
$appStage = Join-Path $taskRoot ('work\package_app_' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $appStage -Force | Out-Null
foreach ($fileName in @('atm_er2_gui.exe','atm_er2_optimizer.exe','nvrtc64_120_0.dll','nvrtc-builtins64_129.dll','README.md','OPTI.md','PERFORMANCE.md','LICENSE','Dear-ImGui-LICENSE.txt')) {
    Copy-Item -LiteralPath (Join-Path $outputDir $fileName) -Destination $appStage
}
Compress-Archive -Path (Join-Path $appStage '*') -DestinationPath (Join-Path $outputDir 'atm-reactor-optimizer-windows.zip') -Force
Write-Output "Packaged executable, NVIDIA runtime, documentation, and source in $outputDir"

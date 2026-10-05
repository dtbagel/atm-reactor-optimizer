param([switch]$Native,[switch]$CliOnly)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$localTools = Join-Path $PSScriptRoot 'local'
$compilerDir = Get-ChildItem -LiteralPath $localTools -Directory -Filter 'llvm-mingw*' | Select-Object -First 1
if (-not $compilerDir) { throw 'Run tools/bootstrap.py using Python first.' }
$compiler = Join-Path $compilerDir.FullName 'bin\clang++.exe'
$buildDir = Join-Path $taskRoot 'build'
New-Item -ItemType Directory -Path $buildDir -Force | Out-Null
$kernel = [IO.File]::ReadAllText((Join-Path $taskRoot 'src\reactor_core.hpp')) + "`n" + [IO.File]::ReadAllText((Join-Path $taskRoot 'src\gpu_kernel.cu'))
$embedded = '#pragma once' + "`n" + 'inline constexpr const char* kernel_source = R"ER2KERNEL(' + $kernel + ')ER2KERNEL";' + "`n"
[IO.File]::WriteAllText((Join-Path $buildDir 'embedded_kernel.hpp'), $embedded)
$arguments = @('-std=c++20','-O3','-Wall','-Wextra','-Wpedantic','-ffp-contract=off','-static',"-I$buildDir")
if ($Native) { $arguments += '-march=native' }
$arguments += @((Join-Path $taskRoot 'src\main.cpp'),(Join-Path $taskRoot 'src\simulator.cpp'),(Join-Path $taskRoot 'src\gpu.cpp'),'-o',(Join-Path $buildDir 'atm_er2_optimizer.exe'))
& $compiler @arguments
if ($LASTEXITCODE -ne 0) { throw 'C++ compilation failed' }
$rtcDir = Join-Path $localTools 'nvidia\cuda_nvrtc\bin'
if (Test-Path -LiteralPath $rtcDir) { Get-ChildItem -LiteralPath $rtcDir -Filter '*.dll' | Copy-Item -Destination $buildDir }
Write-Output "Built $buildDir\atm_er2_optimizer.exe"
if (-not $CliOnly) {
    $imguiDir = Join-Path $taskRoot 'vendor\imgui'
    $guiArguments = @('-std=c++20','-O3','-Wall','-Wextra','-ffp-contract=off','-static','-municode','-mwindows','-DER2_NO_MAIN',"-I$buildDir","-I$imguiDir")
    if ($Native) { $guiArguments += '-march=native' }
    $guiArguments += @((Join-Path $taskRoot 'src\gui.cpp'),(Join-Path $taskRoot 'src\main.cpp'),(Join-Path $taskRoot 'src\simulator.cpp'),(Join-Path $taskRoot 'src\gpu.cpp'))
    foreach ($fileName in @('imgui.cpp','imgui_draw.cpp','imgui_tables.cpp','imgui_widgets.cpp','backends\imgui_impl_win32.cpp','backends\imgui_impl_dx11.cpp')) {
        $guiArguments += (Join-Path $imguiDir $fileName)
    }
    $guiArguments += @('-ld3d11','-ld3dcompiler','-ldwmapi','-lole32','-luuid','-lshell32','-lwindowscodecs','-lgdi32','-limm32','-o',(Join-Path $buildDir 'atm_er2_gui.exe'))
    & $compiler @guiArguments
    if ($LASTEXITCODE -ne 0) { throw 'GUI compilation failed' }
    Write-Output "Built $buildDir\atm_er2_gui.exe"
}

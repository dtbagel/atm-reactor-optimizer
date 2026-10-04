param([switch]$Native)
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

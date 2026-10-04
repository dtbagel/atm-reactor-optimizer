param([int]$Threads = 24, [int]$Layouts = 16384)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $taskRoot 'build\atm_er2_optimizer.exe'
foreach ($batchSize in @(2048,4096,8192,16384,32768,65536,131072)) {
    Write-Output "GPU batch size: $batchSize"
    & $exe --backend cuda --threads $Threads --batch $batchSize --evaluations $Layouts --seed 1337 --min-power 350000 --quiet --output (Join-Path $taskRoot "work\profile_$batchSize")
    if ($LASTEXITCODE -ne 0) { throw 'Profiling run failed' }
}

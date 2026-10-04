# ATM10 Extreme Reactors optimizer — C++20 + CUDA

A working GPU-accelerated port of the supplied `ato.py`. It searches the 49-cell fuel-column layout of a 9×9×9 passive reactor with a 7×7×7 interior and Unobtainium moderator. The default objective is FE/mB, with an optional minimum FE/t.

The NVIDIA backend evaluates thousands of independent layouts together. CPU workers generate candidates in parallel and re-evaluate the final 128 distinct elites with double-precision calculations. The saved winner must pass the power floor after this final evaluation.

## Run the ready-built Windows version

From this project's directory in PowerShell:

```powershell
.\outputs\atm_er2_optimizer.exe --backend cuda --threads 24 --seconds 60 --min-power 350000 --output outputs/best_reactor
```

Keep `nvrtc64_120_0.dll` and `nvrtc-builtins64_129.dll` beside the executable. This uses the installed NVIDIA driver and NVIDIA's runtime compiler; a full CUDA Toolkit installation is unnecessary. `--backend auto` selects CUDA when available and otherwise reports the problem and uses CPU. `--backend cuda` fails clearly if CUDA cannot start.

Remove `--min-power` for maximum efficiency without a power floor. Saved results are `best_reactor.txt` and `best_reactor.json` at the chosen output prefix. `R` is a fuel column; `U` is Unobtainium. Every column is seven blocks tall. The example result bundled here used a 350,000 FE/t floor.

## Useful commands

```powershell
# Validate the checkerboard with the exact CPU reference and the GPU.
.\outputs\atm_er2_optimizer.exe --calibrate --output outputs/checkerboard

# Numerical agreement and ranking checks across 1,024 layouts.
.\outputs\atm_er2_optimizer.exe --backend cuda --self-test --validation-layouts 1024

# CPU/GPU timings followed by a five-second search.
.\outputs\atm_er2_optimizer.exe --backend cuda --benchmark --threads 24 --seconds 5 --min-power 350000

# CPU reference search.
.\outputs\atm_er2_optimizer.exe --backend cpu --math exact --threads 24 --seconds 60

# Repeatable search budget; overrides --seconds.
.\outputs\atm_er2_optimizer.exe --backend cuda --threads 24 --evaluations 1000000 --seed 1337 --min-power 350000

# Inspect a layout mask using the exact CPU equations.
.\outputs\atm_er2_optimizer.exe --backend cpu --math exact --evaluate 0x1555555555555 --output outputs/inspection
```

`--help` lists all options. Settings inherited from Python include `--min-rods`, `--max-rods`, `--insertion` (percent), `--fill`, `--variant-efficiency`, `--search-max-ticks`, `--search-min-ticks`, `--sample-ticks`, `--final-ticks`, and `--seed`. New options include `--backend`, `--math`, `--batch`, `--device`, `--objective efficiency|power`, `--evaluations`, `--progress-ms`, and `--quiet`.

The default batch is 65,536 layouts. `--batch 131072` can help on faster GPUs. Smaller batches trade throughput for shorter launches. The search reports total evaluation count, evaluations/s, simulated ticks/s, and champion improvements/s. These counts include repeated masks; they are not counts of unique layouts.

## Simulation and validation

- One fuel-block irradiation event is processed per tick. There is no multiplication of event frequency by rod count or height. The source cycles through horizontal columns as in `ato.py`; repeating seven identical vertical slices preserves this event sequence for the modeled geometry.
- `--math fast` uses float state and native GPU transcendental functions for discovery. `--math exact` uses double calculations. Double calculations on consumer NVIDIA GPUs are usually much slower; float discovery followed by exact CPU verification is recommended.
- GPU discovery runs the full configured tick budget, default 4,500 ticks, and averages the last 500 ticks. It uses scalar accumulators rather than a large per-thread ring buffer. It does not claim that every candidate has reached steady state.
- The exact CPU evaluator retains Python's adaptive convergence checks and circular sampling window. Final candidates use 20,000 maximum ticks by default and at least a 1,000-tick sample window when the final budget permits it. `--fixed-ticks` disables adaptive stopping for the CPU path.
- Before every run, a built-in CPU check compares the checkerboard against the Python measurement. Accelerated searches also check their checkerboard against the CPU calculation before searching. `--self-test` performs the broader numerical and ranking comparison.
- An infeasible power target returns exit code 3 and saves no new winner. Invalid input and CUDA failures return exit code 1. Existing output files are not removed by an unsuccessful run.

The double C++ checkerboard gives about **529,645 FE/t and 0.873485 mB/t**, versus Python's **529,609 FE/t and 0.873500 mB/t**. The existing thermal integration oscillates; minute floating-point differences change its final phase. Comparisons therefore use average power and fuel rather than requiring identical instantaneous temperatures.

The supplied Python script is the compatibility reference. It assumes maintained fuel fill, standard fuel properties, passive mode, and fixed geometry. Uniform nonzero rod insertion preserves Python's formula: it reduces raw radiation before exponentiation as well as reducing the final scaled value. The [ER2 1.21 irradiation source](https://github.com/ZeroNoRyouki/ExtremeReactors2/blob/1.21/src/main/java/it/zerono/mods/extremereactors/gamecontent/multiblock/reactor/ReactorLogic.java) applies its source insertion modifier after exponentiation. The default 0% insertion is unaffected by that difference. The [source iterator](https://github.com/ZeroNoRyouki/ExtremeReactors2/blob/1.21/src/main/java/it/zerono/mods/extremereactors/gamecontent/multiblock/reactor/FuelRodsMap.java) and [fuel moderation implementation](https://github.com/ZeroNoRyouki/ExtremeReactors2/blob/1.21/src/main/java/it/zerono/mods/extremereactors/gamecontent/multiblock/reactor/part/ReactorFuelRodEntity.java) were also checked. This is not a complete Minecraft world simulator.

The search uses known seeds, distinct elites, mutation, crossover, and random injection. It is heuristic and does not prove global optimality. Fixed-budget runs are repeatable with the same binary, hardware, backend, thread count, batch size, seed, and settings. Time-budget runs are inherently variable. `--seconds` covers search time; startup compilation, calibration, and final verification add a little time, and the last batch can overrun the budget. CUDA compilation occurs once per process.

## Build

The already downloaded compiler and runtime are local to `tools/local/`. They did not install system-wide software. To rebuild on this machine:

```powershell
.\tools\build.ps1 -Native
```

On a fresh Windows machine with Python, first run:

```powershell
python tools/bootstrap.py
.\tools\build.ps1 -Native
```

The bootstrap downloads a portable LLVM-MinGW compiler, CMake, Ninja, and NVIDIA's official `nvidia-cuda-nvrtc-cu12` wheel. It requires internet access. The executable is built into `build/`, with the runtime DLLs copied beside it. `-Native` tunes CPU code for the build machine; omit it for a more portable CPU binary.

For an existing C++20 toolchain and Ninja:

```text
cmake --preset release-native
cmake --build --preset release-native
ctest --test-dir build-native --output-on-failure
```

A regular `cmake -S . -B build` also supports Visual Studio generators. CMake embeds the shared kernel source into the binary, so runtime CUDA compilation requires no source files and no `nvcc`. Put NVRTC beside the executable, in `CUDA_PATH/bin`, on the library search path, or set `ER2_NVRTC_PATH` to its full library path. The driver/NVRTC loader also has a Linux implementation, but Linux was not tested here.

CPU flags used here: Clang 23.1.2, C++20, `-O3 -march=native -ffp-contract=off -static`. No CPU `-ffast-math` is used. CUDA uses native float math intrinsics, targets the detected GPU architecture, and disables fused multiply-add to keep operation ordering stable. The [NVRTC documentation](https://docs.nvidia.com/cuda/nvrtc/) describes runtime compilation and loading PTX through the CUDA driver API.

## Test and profile

```powershell
python tests/regression.py --gpu
.\tools\profile.ps1 -Layouts 524288
```

The regression checks 24 layouts against Python, including boundary and dense cases, different insertion/fill settings, deterministic runs, infeasible targets, CLI bounds, JSON output, a GPU search, and the double GPU path. The GPU self-test checks numerical error, rank correlation, and top-16 agreement. CMake's built-in test runs the CPU reference and float validation.

The profile script compares batch sizes. Search output separates candidate generation from simulation/transfers. See `outputs/benchmark_report.md` for actual measurements on the Ryzen 9 5900X and RTX 3080 Ti. Benchmarks measure complete candidate simulations; performance depends on rod density, tick budgets, batch size, GPU clocks, and other machine load.

## Files

`ato.py` remains the original reference. `src/reactor_core.hpp` contains shared equations and topology, `src/simulator.cpp` provides adaptive CPU sampling, `src/gpu.cpp` loads CUDA/NVRTC and submits GPU batches, and `src/main.cpp` supplies worker threads, evolutionary search, CLI, validation, benchmarks, and result files. The evaluator performs no heap allocations; host batch arrays and GPU buffers are reused.

# ATM10 Extreme Reactors optimizer — C++20 + CUDA

A GPU-accelerated C++ optimizer with a native Dear ImGui interface, custom interior dimensions, and all 68 moderator presets from the supplied ATM10 dataset. It searches fuel-column layouts for a passive reactor. The default objective is FE/mB, with an optional minimum FE/t. The original 7×7×7 Unobtainium setup remains the default.

The NVIDIA backend evaluates thousands of independent layouts together. CPU workers generate candidates in parallel and re-evaluate the final 128 distinct elites with double-precision calculations. The saved winner must pass the power floor after this final evaluation.

The optimized build measured **5.12 million complete simulations/sec** on mixed-layout GPU batches and about **2.55 million/sec** in the full search on the RTX 3080 Ti / Ryzen 9 5900X. Every discovery simulation still runs 4,500 ticks; cached results are excluded from this rate. See [PERFORMANCE.md](PERFORMANCE.md) for controlled before/after measurements, accuracy, and variation. [OPTI.md](OPTI.md) preserves the completed experiments and remaining optimization ideas.

## Native Windows interface

Extract `atm-reactor-optimizer-windows.zip` and open **atm_er2_gui.exe**. Keep both NVIDIA runtime DLLs in the same folder. The interface uses Dear ImGui v1.92.5 with Win32 and DirectX 11, translucent dark glass panels, cyan/purple lighting, and Windows fonts. It does not need a browser or a console window.

1. Choose Auto, NVIDIA GPU, or CPU; set workers and search time in seconds.
2. Enter a minimum power target (0 means no floor).
3. Set **width, depth, and height of the interior**, excluding casing. The outside dimensions are two blocks larger along each axis.
4. Search the moderator list and select a material. One selected material fills every non-fuel column; mixed-material layouts are not modeled.
5. Choose efficiency or power, select an output folder, and run.

The reactor preview shows the current search leader and then the CPU-verified winner. Hover over a cell for its coordinates and material. Fuel columns repeat through the chosen height. The UI distinguishes search estimates from verified results, displays evaluations/sec, and can copy the layout or open the saved results folder. **Finish early & verify** ends the search after the current batch and verifies the best available candidates. Closing the window during a run also finishes the current work before exiting.

**Advanced settings → Search agents & mutations** controls independent populations, power scouts, mutation chance, flips per mutation, rod moves, crossover, random injection, parents per agent, sharing, and automatic restarts. Balanced, Explore broadly, Refine layouts, and Find power presets provide starting points. These settings stay adjustable during a run and apply after the current batch. **Reseed agents** starts fresh populations while preserving the best layouts already found. Increasing agents divides the same evaluation budget among more populations; it does not multiply GPU capacity.

The **Reactor & runtime** tab includes minimum fuel columns, insertion, fuel fill, precision, batch size, GPU device, seed, and result filename. Reactor and runtime settings are locked during a job. **Benchmark** measures the selected geometry/material and opens its timing log. The optimizer runs on a worker thread while DirectX renders the interface.

Supported interior dimensions are **1–32 width, 1–32 depth, and 1–64 height** (up to 1,024 columns). These are this optimizer's input limits, not a claim about the construction limits of every installed game configuration. Every interior position is available for fuel: a 7×7 footprint allows all 49 columns. There is no additional upper fuel-column cap in the GUI. Each axis, reactor volume, surface area, cooling, fuel capacity, radiation boundaries, and heat-transfer geometry feed the CPU and CUDA calculations.

A best layout can plateau while evaluations continue. The interface shows the exact evaluation count, time since the last improvement, population restarts, and the remaining power gap. If a batch takes longer than two seconds, it shows the last-batch age and current activity. Power scouts seek output while the other agents optimize your chosen objective. Try Explore broadly and reseeding when progress plateaus, or Find power when output is the priority. Final verification still enforces your minimum strictly. A heuristic search cannot guarantee a particular target or prove the best possible layout.

## Run the command-line Windows version

From this project's directory in PowerShell:

```powershell
.\outputs\atm_er2_optimizer.exe --backend cuda --threads 24 --seconds 60 --min-power 350000 --output outputs/best_reactor
```

Keep `nvrtc64_120_0.dll` and `nvrtc-builtins64_129.dll` beside the executable. This uses the installed NVIDIA driver and NVIDIA's runtime compiler; a full CUDA Toolkit installation is unnecessary. `--backend auto` selects CUDA when available and otherwise reports the problem and uses CPU. `--backend cuda` fails clearly if CUDA cannot start.

Remove `--min-power` for maximum efficiency without a power floor. Saved results are `best_reactor.txt` and `best_reactor.json` at the chosen output prefix. `R` is a fuel column; `U` is Unobtainium and `M` is another selected moderator. Every column has the selected interior height. The example result bundled here used a 350,000 FE/t floor.

## Useful commands

```powershell
# Validate the checkerboard with the exact CPU reference and the GPU.
.\outputs\atm_er2_optimizer.exe --calibrate --output outputs/checkerboard

# Numerical agreement and ranking checks across 1,024 layouts.
.\outputs\atm_er2_optimizer.exe --backend cuda --self-test --validation-layouts 1024

# CPU/GPU timings followed by a five-second search.
.\outputs\atm_er2_optimizer.exe --backend cuda --benchmark --threads 24 --seconds 5 --min-power 350000

# Custom dimensions and moderator; dimensions are width, depth, height.
.\outputs\atm_er2_optimizer.exe --width 9 --depth 8 --height 12 --moderator graphite --max-rods 50 --seconds 60

# List the complete supplied moderator table (including flowing states).
.\outputs\atm_er2_optimizer.exe --list-moderators

# Benchmark without starting a search.
.\outputs\atm_er2_optimizer.exe --benchmark-only --moderator water --width 5 --depth 3 --height 8

# CPU reference search.
.\outputs\atm_er2_optimizer.exe --backend cpu --math exact --threads 24 --seconds 60

# Repeatable search budget; overrides --seconds.
.\outputs\atm_er2_optimizer.exe --backend cuda --threads 24 --evaluations 1000000 --seed 1337 --min-power 350000

# Broader exploration with independent populations and wider mutations.
.\outputs\atm_er2_optimizer.exe --min-power 350000 --agents 8 --power-agents 2 --mutation-percent 100 --max-flips 12 --random-percent 35 --restart-generations 60

# Inspect a layout mask using the exact CPU equations.
.\outputs\atm_er2_optimizer.exe --backend cpu --math exact --evaluate 0x1555555555555 --output outputs/inspection
```

`--help` lists all options. Settings inherited from Python include `--min-rods`, `--max-rods`, `--insertion` (percent), `--fill`, `--variant-efficiency`, `--search-max-ticks`, `--search-min-ticks`, `--sample-ticks`, `--final-ticks`, and `--seed`. New options include `--width`, `--depth`, `--height`, `--moderator`, `--list-moderators`, `--benchmark-only`, `--backend`, `--math`, `--batch`, `--device`, `--objective efficiency|power`, `--evaluations`, `--progress-ms`, and `--quiet`.

The default batch is 65,536 proposals. `--batch 131072` can help isolated GPU batches, but it was slower in our full-search comparison. Smaller batches trade throughput for shorter launches. The search reports complete simulations, simulations/s, simulated ticks/s, proposals, exact score reuses, agent count, restarts, and time since improvement. A bounded same-run cache and within-batch deduplication avoid re-simulating repeated masks. `--cache-mb 0` disables the persistent cache; `--no-dedup` disables batch deduplication. The default cache budget is 32 MiB. These are not lifetime unique-layout counts: an evicted layout can be simulated again.

`--evaluations` retains its meaning as the proposal budget. JSON `candidate_evaluations` counts proposals, `simulation_evaluations` counts newly simulated candidates, and `cache_hits` counts exact reuses, including `batch_duplicates`. `layouts_per_second` and reactor ticks count only simulations. Search controls also have CLI options: `--agents`, `--power-agents`, `--elites`, `--mutation-percent`, `--max-flips`, `--move-percent`, `--crossover-percent`, `--random-percent`, `--migration-generations`, and `--restart-generations`. Set either generation interval to 0 to disable it. The optional CLI `--max-rods` remains available for explicitly constrained searches; its default is the complete interior footprint.

## Simulation and validation

- One fuel-block irradiation event is processed per tick. There is no multiplication of event frequency by rod count or height. The source cycles through horizontal columns as in `ato.py`; repeating identical vertical slices through the selected height preserves this event sequence for the modeled geometry.
- `--math fast` uses float state and native GPU transcendental functions, FMA, and reciprocal divisors for discovery. `--math exact` uses double calculations, with GPU FMA/reciprocals off by default. Double calculations on consumer NVIDIA GPUs are usually much slower; float discovery followed by exact CPU verification is recommended.
- GPU discovery runs the full configured tick budget, default 4,500 ticks, and averages the last 500 ticks. It uses scalar accumulators rather than a large per-thread ring buffer. It does not claim that every candidate has reached steady state.
- The exact CPU evaluator retains Python's adaptive convergence checks and circular sampling window. Final candidates use 20,000 maximum ticks by default and at least a 1,000-tick sample window when the final budget permits it. `--fixed-ticks` disables adaptive stopping for the CPU path.
- Before every run, a built-in CPU check compares the checkerboard against the Python measurement. Accelerated searches also check their checkerboard against the CPU calculation before searching. `--self-test` performs the broader numerical and ranking comparison.
- An infeasible power target returns exit code 3 and saves no new winner. Invalid input and CUDA failures return exit code 1. Existing output files are not removed by an unsuccessful run.

Float GPU discovery retains estimates up to 0.2% below the target to reduce premature rejection from rounding near the floor. `--discovery-power-slack 0` disables this margin. Displayed feasibility and final CPU acceptance still use the requested minimum; the saved result never receives this allowance.

The double C++ checkerboard gives about **529,645 FE/t and 0.873485 mB/t**, versus Python's **529,609 FE/t and 0.873500 mB/t**. The existing thermal integration oscillates; minute floating-point differences change its final phase. Comparisons therefore use average power and fuel rather than requiring identical instantaneous temperatures.

The supplied Python script is the compatibility reference. Its maintained fuel fill, standard fuel properties, passive mode, and event schedule are preserved. C++ generalizes its geometry formulas and replaces the fixed Unobtainium coefficients with the selected preset. Uniform nonzero rod insertion preserves Python's formula: it reduces raw radiation before exponentiation as well as reducing the final scaled value. The [ER2 1.21 irradiation source](https://github.com/ZeroNoRyouki/ExtremeReactors2/blob/1.21/src/main/java/it/zerono/mods/extremereactors/gamecontent/multiblock/reactor/ReactorLogic.java) applies its source insertion modifier after exponentiation. The default 0% insertion is unaffected by that difference. The [source iterator](https://github.com/ZeroNoRyouki/ExtremeReactors2/blob/1.21/src/main/java/it/zerono/mods/extremereactors/gamecontent/multiblock/reactor/FuelRodsMap.java) and [fuel moderation implementation](https://github.com/ZeroNoRyouki/ExtremeReactors2/blob/1.21/src/main/java/it/zerono/mods/extremereactors/gamecontent/multiblock/reactor/part/ReactorFuelRodEntity.java) were also checked. This is not a complete Minecraft world simulator.

The search uses known seeds, distinct elites, mutation, crossover, and random injection. Each agent keeps its own parents and random generator. Periodic sharing passes champions to neighboring populations. Stagnant populations restart without erasing the global finalist archive. Fixed-budget runs without live intervention are repeatable with the same binary, hardware, backend, thread count, batch size, seed, and settings. Time-budget runs and runs with live tuning are inherently variable. `--seconds` covers search time; startup compilation, calibration, and final verification add a little time, and the last batch can overrun the budget. CUDA compilation occurs once per run and specializes footprint storage, geometry, moderator, insertion, fill, and variant. The loader rejects incompatible evaluation settings. Pinned buffers and a private CUDA stream transfer compact masks and scores; stable rod-count grouping preserves proposal order. Larger interiors or double GPU calculations start with smaller batches to help keep launches brief.

## Moderator data and saved layouts

`data/moderators.json` preserves the supplied schema-2 dataset. All 68 entries are available through `--moderator <key>`; the normal UI lists the 56 entries marked placeable. The 12 flowing fluid states remain in the dataset and CLI for completeness. The app uses absorption, heat efficiency, moderation, and heat conductivity directly. Air is a valid preset.

`src/moderators.hpp` is the bundled table generated from that JSON. To update the data, edit the JSON, run `python tools/generate_presets.py`, then rebuild. No external preset file is required to launch the packaged app. The original `ato.py` is unchanged.

Saved JSON includes the moderator key/name and all four coefficients, `[width, depth, height]`, rod blocks, simulation settings, and the layout. It records the final search tuning, generation/restart/improvement counts, and whether live tuning changed. For footprints above 64 cells, `result.mask` is a hexadecimal string; for smaller footprints it stays a JSON integer for compatibility. `--evaluate` accepts arbitrarily long hexadecimal masks up to the supported footprint. Bits proceed from left to right across each row, then from front to back through the depth.

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

The bootstrap downloads a portable LLVM-MinGW compiler, CMake, Ninja, and NVIDIA's official `nvidia-cuda-nvrtc-cu12` wheel. It requires internet access. Both the CLI and GUI executables are built into `build/`, with the runtime DLLs copied beside it. `-Native` tunes CPU code for the build machine; omit it for a more portable CPU binary.

Use `-CliOnly` to skip the GUI in the PowerShell build. Dear ImGui source and its MIT license are included in `vendor/imgui`; no UI dependency download is needed. CMake builds the GUI on Windows by default; pass `-DER2_GUI=OFF` to build only the CLI.

For an existing C++20 toolchain and Ninja:

```text
cmake --preset release-native
cmake --build --preset release-native
ctest --test-dir build-native --output-on-failure
```

A regular `cmake -S . -B build` also supports Visual Studio generators. CMake embeds the shared kernel source into the binary, so runtime CUDA compilation requires no source files and no `nvcc`. Put NVRTC beside the executable, in `CUDA_PATH/bin`, on the library search path, or set `ER2_NVRTC_PATH` to its full library path. The driver/NVRTC loader also has a Linux implementation, but Linux was not tested here.

CPU flags used here: Clang 23.1.2, C++20, `-O3 -march=native -ffp-contract=off -static`. No CPU `-ffast-math` is used. CUDA uses native float math intrinsics and targets the detected GPU architecture. Float discovery enables fused multiply-add and reciprocal divisors; set `ER2_GPU_FMAD=0` and `ER2_GPU_RECIPROCAL_DIVISORS=0` to preserve the earlier float arithmetic ordering. These defaults and GPU resource diagnostics are saved in result JSON. CPU final verification always keeps reference arithmetic. The [NVRTC documentation](https://docs.nvidia.com/cuda/nvrtc/) describes runtime compilation and loading PTX through the CUDA driver API.

## Test and profile

```powershell
python tests/regression.py --gpu
python tests/materials_dimensions.py --gpu
.\tools\profile.ps1 -Layouts 524288
```

The regression checks 24 layouts against Python, including boundary and dense cases, different insertion/fill settings, deterministic runs, infeasible targets, CLI bounds, JSON counters, strict final acceptance despite discovery slack, a GPU search, and the double GPU path. The additional test compares every one of the 68 moderator presets against the independently generalized Python equations over eight shapes, including 32×32×64, checks masks beyond bit 63, performs CPU/GPU searches, and rejects invalid dimensions. The GPU self-test checks numerical error, rank correlation, and top-16 agreement. Six CTests also check warmup/branchless equivalence, cache collisions and reconstruction, live controls, early finish, and the GUI.

The profile script compares batch sizes. Search output separates generation, simulation/transfers, reuse/deduplication, and selection. CMake also builds `er2_performance` for deterministic full-tick fixtures (`er2_performance.exe 65536 11`) and `gpu_accuracy_test` for the broad matrix (`gpu_accuracy_test.exe --double-stride 17`). Set `ER2_GPU_PROFILE=1` to collect CUDA event and staging timings. See [PERFORMANCE.md](PERFORMANCE.md) for current measurements. Performance depends on rod density, tick budgets, batch size, GPU clocks, and other machine load.

The GUI also has hidden developer checks: `atm_er2_gui.exe --smoke-test <report.txt>` activates Run with ImGui mouse events, renders while running a custom 3×5×4 water search, and verifies the saved winner. `--target-test <report.txt>` tests the actual GPU UI path with a 7×7×7 Unobtainium reactor and a 350,000 FE/t floor. `--screenshot <preview.png>` captures the actual DirectX back buffer; add `--demo` for a short live run before capture or `--advanced-preview` to show the tuning controls. These are native UI renders. CTest also checks live tuning, agent resizing, manual/automatic reseeding, archive preservation, early finish, and the exact evaluation budget.

The older `outputs/benchmark_report.md` describes the original fixed-geometry baseline; OPTI.md labels those numbers as historical and records the current pass separately. The current GUI target test completed 12,697,473 actual simulations and reused 7,749,759 scores during its five-second search, saving an exact CPU-verified 352,387.61 FE/t layout with eight fuel columns. This confirms the requested 350,000 target for this modeled setup; it is not a claim of global optimality.

## Files

`ato.py` remains the original reference. `src/reactor_core.hpp` contains shared equations and topology, `src/simulator.cpp` provides adaptive CPU sampling, `src/gpu.cpp` loads CUDA/NVRTC and submits GPU batches, and `src/main.cpp` supplies worker threads, evolutionary search, CLI, validation, benchmarks, and result files. `src/gui.cpp` supplies the Windows interface, `src/engine.hpp` its worker callbacks, and `src/moderators.hpp` the generated preset table. The evaluator performs no heap allocations; host batch arrays and GPU buffers are reused.

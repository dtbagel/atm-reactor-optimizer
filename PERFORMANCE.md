# Optimization measurements — October 4, 2026

The revised optimizer measured **2.55 million actual reactor simulations/sec in complete searches**, versus **1.50 million/sec before**: about **70% more** on this machine. Controlled mixed-layout GPU batches improved from **1.93 million to 5.12 million/sec**. Reused scores are excluded from both simulation rates.

Every GPU discovery simulation still performs 4,500 reactor ticks and averages the final 500. The source schedule, maintained fuel fill, and original exact CPU verification remain intact. The modeled eight-column winner produced **352,387.607 FE/t**, above the user's 350,000 FE/t target, at 0.281708469 mB/t and 1,250,894.613 FE/mB. This is a heuristic result, not a global optimality claim.

## Hardware and build

- Ryzen 9 5900X, 24 logical CPU workers; RTX 3080 Ti 12 GB, compute capability 8.6.
- Windows display GPU, NVIDIA driver 596.36, NVRTC 12.9.86.
- LLVM-MinGW Clang 23.1.2, C++20, native Release build; CPU `-O3 -march=native -ffp-contract=off -static`, without fast-math.
- Before: preserved GUI/multi-agent build at commit `2222a5df70e49f6c78ea3dd145244e71dc400b35`. After: current optimized local source.
- Default geometry/material: 7×7×7 Unobtainium, zero insertion, full fill, all 49 columns permitted. GPU batch 65,536; 128 threads/block.

## Controlled identical-layout batches

Three alternating before/after rounds each used 11 timed repetitions after warmup. Masks come from identical deterministic fixtures, with seed 1234567. Compilation and fixture generation are outside the timer. Host packing, transfers, full GPU simulation, synchronization, and result reconstruction are inside. The table uses the median of the three per-round median rates, rather than the fastest run.

| Rod columns | Before, million simulations/s | After, million simulations/s | Ratio |
|---|---:|---:|---:|
| 1 | 2.649 | 4.685 | 1.77× |
| 8 | 2.355 | 5.208 | 2.21× |
| 25 | 2.436 | 5.579 | 2.29× |
| 49 | 3.280 | 7.166 | 2.18× |
| Uniformly mixed 1–49 | 1.928 | 5.119 | 2.66× |

Mixed-case per-round medians were 2.030, 1.843, 1.928 million/s before and 5.212, 3.831, 5.119 million/s after. Shared display-GPU activity caused visible variation. These gains depend on GPU, shape, density, numerical mode, and machine load; the older fixed-geometry 2.63 million/s search is a different historical workload.

## Full optimizer with the power target

Six searches used three seeds, ten search seconds each, 24 CPU workers, the same 350,000 FE/t floor, default four agents and one power scout. Before/after order was reversed for seed 77. Search time includes candidate generation, dedup/cache work, simulation/transfers, and selection. Startup compilation/calibration and final CPU verification are outside that timer. The last batch can slightly overrun ten seconds.

| Seed | Before, million simulations/s | After, million simulations/s | After simulations | After exact reuses |
|---|---:|---:|---:|---:|
| 1337 | 1.651 | 2.563 | 25,643,189 | 15,644,491 |
| 77 | 1.468 | 2.547 | 25,481,282 | 15,478,718 |
| 919 | 1.500 | 2.421 | 24,218,078 | 14,702,886 |
| Median rate | **1.500** | **2.547** | — | — |

The median actual simulation rate increased **1.70×**. The median proposal rate was about 4.095 million/s; roughly 38% of proposals reused an exact same-run score. The baseline had no score reuse, so its proposals were simulations. All six searches saved the same verified `0x2c28b0000` winner. Because generation and arithmetic changed, these evolving populations are not identical-layout fixtures; this measures useful pipeline capacity and observed outcome, not universal search-quality improvement.

A separate after run completed 25,564,361 simulations and 41,156,608 proposals in 10.000379 seconds. Its timer breakdown was:

| Work | Seconds | Share of search time |
|---|---:|---:|
| Generation | 1.111 | 11.1% |
| Simulation/transfers | 5.810 | 58.1% |
| Reuse/deduplication | 2.266 | 22.7% |
| Selection | 0.784 | 7.8% |
| Other | 0.028 | 0.3% |

This makes host reuse/feeding overhead a meaningful next target. An illustrative eight-rod batch with CUDA event profiling measured 11.919 ms kernel, 0.065 ms upload, 0.122 ms download, 0.322 ms packing, and 0.725 ms reconstruction. The selected kernel reported 39 registers and 208 local bytes/thread. This is event timing, not an instruction/stall analysis.

## What changed

The build skips unused warmup observables, specializes GPU settings and footprint storage, uses native popcount, transfers compact masks/results through pinned buffers on a private stream, groups candidates stably by rod count, and reduces divergent ray interpretation. A 7×7 float candidate transfers 8 input / 32 output bytes instead of 128 / 184. Double results use 56 output bytes.

Float discovery enables FMA and reciprocal divisors. Final double CPU arithmetic is unchanged; double GPU defaults leave both disabled. Candidate generation samples minority cells for dense layouts; selection works with compact score/index records and independent workers. Exact bounded same-run reuse and batch deduplication preserve proposal order and fixed proposal budgets. GUI and JSON report simulations and reuses separately.

## Correctness checks

- Warmup and branchless host steps were bit-identical to original branching steps across 71 float/double cases, covering all moderators and geometry/insertion/fill/sample boundaries.
- With FMA/reciprocals disabled, 327,680 GPU fixture results were byte-identical to the original GPU. With selected float arithmetic, maximum power/fuel/efficiency differences against the original float GPU were 0.016391% / 0.068274% / 0.073325%, with complete top-128 overlap and one changed power-floor classification.
- Broad CPU-double comparison: 208 configurations, 26,624 float outputs, all 68 moderators across eight shapes and three tick/sample protocols. Maximum power/fuel/efficiency errors were 0.152975% / 0.001654% / 0.152961%; the worst power error occurred in a deliberately short seven-tick case. Minimum top-32 overlap was 100%; a deliberately ambiguous near-floor fixture changed classification once. Eight double-GPU configurations / 1,024 outputs agreed to displayed precision.
- The ordinary 1,024-layout 4,500-tick self-test had maximum errors 0.002870% power, 0.001654% fuel, and 0.004228% efficiency; rank correlation 0.99999961 and top-16 overlap 16/16.
- Python regression and independently generalized all-moderator/dimension checks passed, including large masks and the double GPU path. Six CTests passed, covering live agents/reseeding, early finish, warmup, cache collisions/evictions/reconstruction/counters, and the GUI.
- The native GPU GUI button/worker test saved the verified 352,387.607 FE/t winner with no ImGui diagnostics. It performed 12,697,473 simulations and 7,749,759 exact reuses in its five-second search.
- A dedicated near-floor rejection test requested 495,807.760566 FE/t from a full layout whose final CPU power was 495,312.448118 FE/t. Discovery retained it within the allowance; final verification returned exit code 3 and saved no winner.

Tiny arithmetic changes can alter the inherited thermal oscillation phase, so averaged power/fuel and ranking matter more than instantaneous final temperatures. Float discovery retains estimates up to 0.2% below the requested floor (`--discovery-power-slack 0.002`), but display feasibility and final CPU acceptance always use the original minimum. `--discovery-power-slack 0` disables the allowance.

## Experiments rejected or left optional

Packed rays and forced unrolling were neutral or slower; both default off. Block sizes 64–1,024 were tested; 128 remained the practical default. Batch 131,072 helped isolated batches but measured only 2.440 million actual simulations/s in a complete search versus 2.556 million/s at 65,536, so the smaller batch remains default.

A GPU temperature-response lookup prototype over 0–20,000 °C had insufficient consistent benefit and regressed the eight-rod case. Its product code was removed; it did not become a validated numerical mode. Analytic ray responses, adaptive convergence, warm starts, GPU-resident evolution, surrogates, and more advanced archive strategies remain future experiments in OPTI.md.

## Reproduce and inspect

Build with CMake, then run from the project root:

```powershell
.\build-cmake\er2_performance.exe 65536 11
.\build-cmake\atm_er2_optimizer.exe --backend cuda --threads 24 --seconds 10 --seed 1337 --min-power 350000 --output work/measurements/result
.\build-cmake\gpu_accuracy_test.exe --double-stride 17
python tests/regression.py --exe build-cmake/atm_er2_optimizer.exe --gpu
python tests/materials_dimensions.py --exe build-cmake/atm_er2_optimizer.exe --gpu
ctest --test-dir build-cmake --output-on-failure
```

Set `ER2_GPU_PROFILE=1` for event timings. Set `ER2_GPU_FMAD=0` and `ER2_GPU_RECIPROCAL_DIVISORS=0` for earlier float arithmetic ordering. `--cache-mb 0 --no-dedup` disables reuse; proposal budgets and actual simulation rates remain distinct. Saved JSON records GPU defaults/resources, proposal/simulation/reuse counts, timer breakdown, and strict CPU verification.

Local raw CSVs, fixture binaries, baseline executables, paired-search JSON/logs, PTX, and the rejected prototype are retained in ignored `work/optimization`. The source archive contains the benchmark harness and validation tests. OPTI.md contains the full brainstorm and next priorities. This report accompanies the optimized source.

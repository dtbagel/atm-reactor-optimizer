# Measured results — October 4, 2026

Hardware: AMD Ryzen 9 5900X (12 cores, 24 logical processors), NVIDIA GeForce RTX 3080 Ti (12 GB), NVIDIA driver 596.36. Windows build: LLVM-MinGW Clang 23.1.2, C++20, `-O3 -march=native -ffp-contract=off -static`. CUDA runtime compiler: NVRTC 12.9.86, targeting compute capability 8.6. GPU discovery uses native float transcendental functions; final verification uses double CPU calculations. No CPU `-ffast-math`.

One evaluation is one complete candidate-layout simulation. Counts include duplicate masks. GPU search simulated all 4,500 ticks of every candidate and averaged the final 500 ticks. The benchmark's fixed-tick CPU/GPU rows used the same 8,192 random layouts and settings.

| Measurement | Layout evaluations/s |
|---|---:|
| C++ double CPU, one worker, fixed ticks | 3,037 |
| C++ double CPU, 24 workers, fixed ticks | 45,850 |
| C++ float CPU, 24 workers, fixed ticks | 52,616 |
| CUDA float, batch 8,192, including transfers | 409,596 |
| Final CUDA search, batch 65,536, 24 CPU workers | **2,627,968** |
| Original Python search, 24 processes | 679 |

The 8,192-layout CUDA microbenchmark was about 8.9× faster than the double C++ run using 24 workers. Larger batches raised throughput substantially. The final ten-second search evaluated **26,279,936 candidates in 10.000099 seconds**, equivalent to **11.826 billion simulated reactor ticks per second**. Candidate generation took 0.182 seconds and simulation/transfers took 9.101 seconds; the remainder included selection and progress reporting. Startup compilation, calibration, and final verification are outside this search timer.

The original Python run tested 3,397 candidates in its five-second worker budgets. Compared with the final C++/CUDA search rate, that is roughly 3,900× more evaluations per second. This is an example end-to-end comparison, not an identical-layout microbenchmark: both searches used the same reactor settings and 350,000 FE/t floor, but their random populations, selection strategies, and convergence behavior differed.

## Accuracy

| Checkerboard reference | Power FE/t | Fuel mB/t |
|---|---:|---:|
| Supplied Python | 529,608.855 | 0.873500373 |
| Double C++ CPU | 529,645.346 | 0.873484614 |
| CUDA float, 20,000 ticks | 529,622.500 | 0.873501420 |

The double C++ averages differ from Python by about 0.0069% in power and 0.0018% in fuel. Instantaneous temperatures can differ because the inherited thermal integration oscillates and is sensitive to rounding.

Across **1,024 layouts**, CUDA float versus fixed-tick double C++ had maximum errors of **0.001727% power**, **0.001724% fuel**, and **0.003370% efficiency**. Rank correlation was **0.99999875**, with **16/16 top-layout overlap**. Tests covered rod counts through 49.

The integration regression also checked 24 layouts against Python, including sparse/dense and boundary cases and changed insertion/fill/variant settings. Maximum errors were **0.001164% power**, **0.000523% fuel**, and **0.000641% efficiency**. Deterministic evaluation budgets, the power floor, invalid inputs, saved JSON, GPU search, and the double GPU path passed. The Windows CMake build and CTest passed.

## Example winner

The final search used 0% insertion, full fuel fill, pure Unobtainium, and a 350,000 FE/t floor. The following winner was re-evaluated with the double CPU backend for 20,000 ticks, using a final 1,000-tick averaging window:

```text
U U U U U U U
U U U U U U U
U U R R U R U
U U R U R U U
U U R R U R U
U U U U U U U
U U U U U U U
```

- Fuel columns: **8** (56 fuel blocks)
- Power: **352,387.607 FE/t**
- Fuel: **0.281708469 mB/t**
- Efficiency: **1,250,894.613 FE/mB**
- Mask: `0x2c28b0000`

`best_reactor.json` records the full settings and measured result. This is the best found by the heuristic search, not a proof of the global optimum.

## Performance choices

Batch-size sweeps showed the cost of underfilling the GPU. Representative 524,288-candidate searches above 350,000 FE/t reached approximately 1.55 million/s at 32,768 layouts, 1.98 million/s at 65,536, and 2.16 million/s at 131,072 before the final host-side improvements. Different batch sizes change how often elites are updated, so their populations are not identical. The default is 65,536.

The final implementation uses a bounded shortlist rather than sorting the whole population, expands that shortlist when duplicates require it, and generates independent candidate chunks on the requested CPU workers. A measured 524,288-candidate run reduced generation time from about 31 milliseconds to 3.9 milliseconds after parallel generation. These are timing measurements, not an Nsight instruction-level profile. SIMD, GPU math lookup tables, approximation of the physics, and extra convergence shortcuts were not needed to reach the measured throughput.

To reproduce the broad comparison, run the packaged executable with `--backend cuda --benchmark --threads 24 --seconds 10 --min-power 350000`. To run the batch sweep, use `tools/profile.ps1 -Layouts 524288`. Results vary with GPU clocks, machine load, seed, and population density.

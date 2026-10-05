# Optimization backlog and handoff

Captured on October 4, 2026, from the optimization brainstorm for the ATM10 Extreme Reactors optimizer.

**Status: first revised optimization pass implemented and validated, October 4, 2026.** The original brainstorm is retained below. The completed experiments, defaults, rejected variants, and remaining work are recorded here so the next pass can resume without repeating them. Preserve the reference simulation and final CPU verification while measuring full simulations, cache reuse, and search outcomes separately.

## Project and baseline

- Repository: [dtbagel/atm-reactor-optimizer](https://github.com/dtbagel/atm-reactor-optimizer).
- Current pre-optimization baseline: commit `2222a5df70e49f6c78ea3dd145244e71dc400b35`, on `main`. The original fixed-geometry benchmark used `9e84387a47baf3597ba4135f427bdc5f1adc75f5`.
- Original compatibility reference: `ato.py`.
- Geometry: user-selected interior, width/depth 1–32 and height 1–64, up to 1,024 full-height columns. The default remains 7×7×7. These are optimizer limits, not asserted game construction limits.
- Moderator: one selected preset per layout, from the supplied 68 entries (56 placeable). Default Unobtainium. Standard Uranium/Yellorium-family fuel.
- Search: independent populations, power scouts, configurable mutations/crossover/random injection, migration, automatic restarts, and manual reseeding are implemented. The full footprint is available for fuel. More agents share the evaluation budget.
- Objective: maximize FE/mB, optionally subject to a minimum FE/t, enforced again after final CPU verification.
- Hardware measured: AMD Ryzen 9 5900X, 12 cores/24 logical processors; NVIDIA GeForce RTX 3080 Ti, 12 GB.
- Build: LLVM-MinGW Clang 23.1.2, C++20, `-O3 -march=native -ffp-contract=off -static`; NVRTC 12.9.86, compute capability 8.6; NVIDIA driver 596.36.
- GPU discovery uses float state and native transcendental functions. Final distinct elites are re-evaluated with the double CPU reference.

### Completed optimization pass

The accepted changes keep all 4,500 discovery ticks and the final 500-tick sample. They do not shorten convergence, replace the simulator with a predictor, or count cached scores as completed simulations.

- Warmup skips reported power and fuel-usage calculations while retaining every heat/fertility update. Original branching CPU arithmetic remains the compatibility path. Native popcount replaces the host bit loop.
- GPU layout storage is compiled to the footprint size; input masks use only active words. Float GPU results carry six floats and two integers, and the host restores the original mask. For 7×7, transfers fall from 128 input / 184 output bytes per candidate to **8 / 32 bytes**. Double output is 56 bytes.
- Compilation specializes geometry, moderator, insertion, fill, and variant to each run; a mismatch is rejected. Pinned staging buffers, a private stream, async transfers, and one completion wait replace blocking pageable transfers. Optional CUDA events separate kernel and transfer time.
- Stable rod-count grouping reduces divergent source cycles, then scatters results back to proposal order. Homogeneous batches skip grouping work. Branchless fuel/moderator responses preserve the length, intensity, absorption, and thermal guards.
- Float discovery enables FMA and reciprocal divisors. These change rounding and are explicitly validated approximations to arithmetic ordering. Double GPU defaults keep both disabled, and final CPU arithmetic is unchanged.
- Bounded exact same-run score reuse and within-batch deduplication skip repeated masks. Fingerprints are checked against every active key word. Default cache budget is 32 MiB (262,144 entries, about 18 MiB at 7×7); it is discarded between runs.
- Dense candidate generation samples the smaller of rods or empty cells. Parent selection computes compact score/index records, selects indices, and runs independent population selection in parallel with reused buffers.
- Float GPU discovery retains candidates within 0.2% below the requested floor. CPU final selection restores the original floor strictly. `--discovery-power-slack 0` disables the margin; it never relaxes saved-result acceptance.
- The GUI rate and JSON `layouts_per_second` count actual simulations. Proposal budgets stay compatible: `candidate_evaluations = simulation_evaluations + cache_hits`. `batch_duplicates` is a subset of those reuses. These are not lifetime unique-layout counts: evictions can lead to re-simulation.

#### Controlled full-tick throughput

Baseline is the preserved binary at `2222a5d`; after is the current native build. Both evaluate identical deterministic 65,536-mask fixtures at 7×7×7, Unobtainium, full fill, zero insertion, 4,500/500 ticks. Three alternating before/after rounds use 11 timed repetitions per case after warmup. Compilation and mask generation are outside the timer; packing, GPU transfers, kernel, synchronization, and reconstruction are inside. Values are the median of the three per-round median rates.

| Fuel columns | Before simulations/s | After simulations/s | Ratio |
|---|---:|---:|---:|
| 1 | 2.649 million | 4.685 million | 1.77× |
| 8 | 2.355 million | 5.208 million | 2.21× |
| 25 | 2.436 million | 5.579 million | 2.29× |
| 49 | 3.280 million | 7.166 million | 2.18× |
| Uniformly mixed 1–49 | 1.928 million | 5.119 million | **2.66×** |

The shared Windows display GPU showed variation: mixed-case round medians ranged 1.843–2.030 million/s before and 3.831–5.212 million/s after. These are machine-specific measurements, not guarantees for other GPUs or dimensions. The best isolated run is not the reported aggregate.

A ten-second complete search at the user's 350,000 FE/t floor processed **25,564,361 actual simulations (2.556 million/s)**, 41,156,608 proposals, and 15,592,247 exact reuses. Generation took 1.111 s, simulation/transfers 5.810 s, reuse/dedup 2.266 s, and selection 0.784 s. Selection fell from 2.480 s in the preceding optimization build to 0.784 s; that comparison is an evolving search, not identical masks. Three paired ten-second searches at seeds 1337, 77, and 919 increased the median actual simulation rate from **1.500 to 2.547 million/s (1.70×)**. All six searches saved the same CPU-verified eight-column winner above the floor. Full data are in [PERFORMANCE.md](PERFORMANCE.md).

#### Accuracy and acceptance checks

- Warmup and branchless state/output comparisons are bit-identical across 71 host float/double fixtures, all 68 moderators, changed geometry/insertion/fill, boundaries, and sample windows.
- The compact/specialized/grouped/branchless GPU path with FMA and reciprocals disabled produced **327,680 byte-identical results** against the original GPU fixture. Enabling the selected float arithmetic changed maximum averaged power/fuel/efficiency by 0.016391% / 0.068274% / 0.073325% against that old float GPU; top-128 overlap stayed complete. One near-floor classification changed.
- Broad selected-path validation compared 26,624 float results over 208 configurations against fixed-tick CPU double: maximum power/fuel/efficiency errors 0.152975% / 0.001654% / 0.152961%, including deliberately short seven-tick fixtures. Every fixture retained full top-32 overlap. One deliberately ambiguous near-floor case changed classification. Eight exact-GPU configurations / 1,024 results agreed to the displayed precision.
- The ordinary 1,024-layout, 4,500-tick self-test had maximum power/fuel/efficiency errors 0.002870% / 0.001654% / 0.004228%; rank correlation 0.99999961 and top-16 overlap 16/16.
- Python regression, all-moderator/custom-dimension checks, six CTests, GPU GUI button/worker target, strict final-floor rejection, cache collisions/evictions, proposal reconstruction, and live controls passed. The GUI test saved 352,387.607038 FE/t; a separate full-layout test rejected 495,807.760566 FE/t when CPU power was only 495,312.448118 FE/t, despite the discovery margin.

#### Tested variants not adopted

- Packed ray words and forced ray-loop unrolling were neutral or slower in useful workloads. Keep their experiment switches, default off.
- Block sizes 64, 128, 256, 512, and 1,024 were tested; 128 retained the best balance. Larger blocks were not consistently faster.
- Batch 131,072 improved isolated batches but reduced the full-search rate from 2.556 to 2.440 million actual simulations/s in the measured run. Keep default 65,536; users can tune it.
- A one-degree GPU temperature-response lookup prototype tabulated hardness, effective radiation, and fuel response over 0–20,000 °C, with interpolation and native fallback. Mixed throughput was about 5.404 million/s versus a nearby native 5.320 million/s, while the eight-rod case regressed from about 5.460 to 4.649 million/s. Variation and added approximation/complexity did not justify adoption. **All lookup code was removed from product sources.** The unvalidated prototype and raw fixtures remain in ignored `work/optimization/lut-prototype` for a future controlled study.

#### Defaults, instrumentation, and next priorities

The default float GPU path uses 128 threads/block, settings specialization, rod grouping, branchless rays, FMA, and reciprocal divisors; packed rays/unrolling are off. `--cache-mb 0 --no-dedup` disables score reuse. To keep original float arithmetic ordering, set `ER2_GPU_FMAD=0` and `ER2_GPU_RECIPROCAL_DIVISORS=0` before launch; final CPU results always use the reference equations.

Developer environment switches also include `ER2_GPU_BLOCK_THREADS`, `ER2_GPU_GROUP_RODS`, `ER2_GPU_BRANCHLESS_RAYS`, `ER2_GPU_PACKED_RAYS`, `ER2_GPU_UNROLL_RAYS`, `ER2_GPU_SPECIALIZE_SETTINGS`, `ER2_GPU_PROFILE`, and optional `ER2_GPU_DUMP_PTX=<path>`. GPU profiling records packing/upload/kernel/download/unpacking, grouping, registers, local memory, and transfer sizes. This is event timing, not an Nsight instruction/stall profile. The measured selected kernel used 39 registers and 208 local bytes/thread.

Next priorities: reduce cache/dedup and GPU feeding overhead; inspect spills and instruction bottlenecks with Nsight; then test compact analytic four-ray response coefficients. Adaptive convergence, warm starts, resident GPU evolution, and surrogates remain unimplemented, higher-risk experiments. Do not infer steady state by stopping on a single oscillation phase. Measure verified quality across multiple seeds alongside throughput. The numerical retention margin can let near-floor layouts compete for the finite 128-candidate archive; reserving slots for candidates above the original discovery floor is a possible future safeguard, while current final CPU acceptance is strict.

### Historical measured throughput

| Measurement | Layout evaluations/s |
|---|---:|
| Double C++ CPU, one worker, fixed ticks | 3,037 |
| Double C++ CPU, 24 workers, fixed ticks | 45,850 |
| Float C++ CPU, 24 workers, fixed ticks | 52,616 |
| Float CUDA, batch 8,192, including transfers | 409,596 |
| Final CUDA search, batch 65,536, 24 CPU workers | **2,627,968** |
| Original Python search, 24 processes | 679 |

The final ten-second run evaluated 26,279,936 candidates in 10.000099 seconds: approximately 11.826 billion simulated reactor ticks/s. Candidate generation took 0.182 seconds, simulation/transfers took 9.101 seconds, and selection/progress/other work accounted for the remainder. Startup compilation, calibration, and final verification are outside the search timer.

**Historical inference only:** simulation/transfers occupied approximately 91% of that run, while generation occupied 1.8%. Eliminating generation alone would yield only about a 1.02× improvement for that workload. These percentages must be remeasured for the current multi-agent search. Simulation/transfers is a combined timer, not a measurement proving that the kernel alone takes 91%.

The newer native GUI target check completed 7,147,301 evaluations in 5.013824 search seconds (1.43 million/sec), and CPU-verified 352,387.607 FE/t at the requested 350,000 floor. This used different populations and allowed all 49 columns. It is not a controlled speed comparison against the historical 2.63 million/sec result.

The 8,192-layout CPU/CUDA fixed-tick benchmark used the same layouts and settings. The Python-versus-CUDA end-to-end comparison used different evolving populations and convergence strategies; it is not an identical-layout microbenchmark.

See [the benchmark report](outputs/benchmark_report.md) for the recorded measurements and [README.md](README.md) for build/run instructions.

### Numerical baseline

| Checkerboard implementation | FE/t | mB/t |
|---|---:|---:|
| Python | 529,608.855 | 0.873500373 |
| Double C++ CPU | 529,645.346 | 0.873484614 |
| Float CUDA, 20,000 ticks | 529,622.500 | 0.873501420 |

Across 1,024 layouts, float CUDA versus fixed-tick double C++ had maximum errors of 0.001727% power, 0.001724% fuel, and 0.003370% efficiency; rank correlation was 0.99999875 and top-16 overlap was 16/16. A separate 24-layout Python regression, including boundary/dense cases and changed insertion/fill settings, passed with maximum errors of 0.001164% power, 0.000523% fuel, and 0.000641% efficiency.

The inherited thermal integration oscillates and is sensitive to tiny rounding differences. Instantaneous temperatures can differ even when average power/fuel agree closely. Do not evaluate proposed improvements using a single final temperature snapshot alone.

The example verified winner above 350,000 FE/t has eight fuel columns, mask `0x2c28b0000`, power 352,387.607 FE/t, fuel 0.281708469 mB/t, and efficiency 1,250,894.613 FE/mB. This is a heuristic-search result, not proof of global optimality.

## Existing architecture to inspect before resuming

- `src/reactor_core.hpp`: shared topology, math, per-tick state update, and fixed-tick evaluator.
- `src/gpu_kernel.cu`: one CUDA thread per candidate, calling the shared fixed-tick evaluator.
- `src/gpu.cpp`: CUDA driver/NVRTC loading, per-run specialization, footprint-sized pinned staging, stable rod grouping, compact results, async stream transfers, one completion wait, diagnostics, and optional event timing. Float GPU FMA/reciprocals default on; double defaults off.
- `src/simulator.cpp`: double CPU evaluator with Python-style adaptive checks and circular sample windows.
- `src/main.cpp`: CPU worker pool, parallel candidate generation, evolutionary search, distinct elites, bounded shortlist selection, CLI, benchmarks, validation, and output.
- `tests/regression.py`: Python parity, deterministic budgets, constraints, CLI, output, GPU search, and double GPU checks.
- `tests/core_warmup.cpp`, `tests/search_cache.cpp`, `tests/gpu_accuracy.cpp`: warmup/ray equivalence, collision/order/counter checks, and broad GPU material/geometry/sample-window validation.
- `tools/performance.cpp`: controlled deterministic GPU fixtures, repeated timings, optional saved result fixtures and event profiling; CMake target `er2_performance`.
- `tools/profile.ps1`: existing batch-size sweep. This is timing-based profiling, not an Nsight instruction-level profile.

GPU discovery currently runs all 4,500 configured ticks for every candidate and averages the final 500. It does not implement adaptive GPU convergence. CPU final verification defaults to 20,000 maximum ticks and a 1,000-tick final sample window. The default GPU batch size is 65,536.

## Correctness boundaries

1. Preserve **one fuel-block irradiation event per tick**. Do not multiply event frequency by rod count or height. Parallelizing the four rays from one event is different from irradiating several sources in the same tick.
2. Preserve the final exact CPU check and reapply the power floor after that check.
3. `ato.py` is the compatibility reference for the current port. Uniform nonzero insertion in Python reduces raw radiation before exponentiation and also reduces the final scaled radiation. ER2's checked 1.21 source applies source insertion after exponentiation. Default 0% insertion is unaffected. Do not silently mix a physics correction with a throughput experiment.
4. Maintained fuel fill is a current model assumption. Fuel burn does not deplete the fuel amount during a candidate simulation. Some optimizations below depend on that assumption and would need revisiting for a depletion simulator.
5. Separate algebraic/implementation improvements from approximations that can affect ranking, convergence, or attractor selection.
6. Only the completed experiments above have measured gains. The retained brainstorm below describes hypotheses and future work unless explicitly covered by the completion record; do not multiply speculative gains together.

## Recommended first experiment sequence

1. Establish a fresh baseline on identical fixed masks, at several densities and shapes, with compilation/generation outside timed simulation. Record transfer/kernel/host times and GPU resource use.
2. Deduplicate before simulation and reuse exact same-run scores. Report proposal count, completed simulations, cache reuse, and unique layouts separately; never count a cache hit as a simulated reactor.
3. Skip warmup observable calculations that do not feed back into state, preserving fertility and heat updates. Inspect the compiled code and measure actual savings.
4. Compact footprint-sized input masks and GPU score outputs. The current host mask is 128 bytes even for 49 cells; returning that mask is redundant. Benchmark packing/unpacking overhead.
5. Test settings specialization and compact ray responses. The 31-pattern bound still holds for any single moderator and four-step rays; coefficients must include the chosen settings.
6. Tune block size, topology accesses, and FMA independently. Treat numerical changes as separate experiments with ranking and near-floor validation.
7. Investigate phase-aware adaptive convergence, warm starts, GPU-resident evolution, and overlap after the simpler experiments are measured.

The two biggest structural bets are **replacing repeated ray interpretation with compact response evaluation** and **replacing thousands of cold-start ticks with a validated convergence or steady-state solution**.

## Full ranked brainstorm

### 1. Stop calculating fuel usage during warmup

**Why it stands out:** `State::step` calculates the fertility modifier and `log10(fertility)` every tick. The fixed GPU evaluator only sums fuel usage during the final sample window. With fixed fuel fill, earlier fuel usage does not influence temperature, radiation, fertility, or later fuel amount.

For the default 4,500/500 configuration, omit fuel-usage computation for the first 4,000 ticks. This can remove up to 4,000 source-level logarithms per candidate; inspect compiled code because unused work may already be eliminated. The low-fertility branch already avoids a logarithm when fertility is at most one. Continue updating fertility itself so the measurement window starts with the same state.

Reported electrical power can likewise be computed only during measurement, but all coolant energy transfers and heat updates must still happen on every tick.

**Experiment:** distinguish state evolution from observable calculation. Compare against the unchanged fixed-tick evaluator on identical masks and settings. This should preserve modeled results under the maintained-fill assumption, apart from any unintended implementation changes.

### 2. Specialize a kernel for default reactor settings

Default 0% insertion, full fill, one moderator, and fixed geometry permit stronger simplification than a general runtime-configured kernel.

- Remove control-rod bonus/penalty arithmetic when insertion is zero.
- Replace moderator divisions with compile-time simplifications where appropriate.
- Eliminate unused settings loads and branches.
- Keep a separate general kernel for non-default parameters.

**Experiment:** use compile-time settings or separate kernel variants; measure instruction count, register use, throughput, and numerical agreement. Changing a divide to multiplication by a reciprocal may change rounding, so classify that separately from removing zero-valued terms.

### 3. Prove guards redundant and compile them out

The normal fuel response and hardness ranges bound scaled fuel absorption below the `min(1, absorption)` cap. Fuel/moderator absorption also cannot remove more than incoming intensity in the valid parameter range.

Potential removals include repeated absorption caps and nonnegative-intensity clamps. Derive whether the tiny-intensity ray cutoff can ever trigger for the specialized default settings; omit it only when a valid bound proves it irrelevant.

**Experiment:** document assumptions and bounds, retain the general guarded path, and compare edge cases. Do not remove thermal/fertility clamps merely because radiation clamps appear redundant.

### 4. Enable GPU fused multiply-add

The baseline compiler uses `--fmad=false`. Allow FMA in discovery and compare operation count/dependency chains with the baseline.

FMA changes rounding. Judge averaged power/fuel, efficiency error, ordering, and power-floor decisions, especially for nearly tied layouts. Preserve the reference path. This is a plausible implementation gain, not a promised multiplier.

### 5. Precompute response functions for the 31 possible ray patterns

A ray has length zero through four, with each encountered cell either fuel or the single selected moderator. There are at most `1 + 2 + 4 + 8 + 16 = 31` patterns, for any supported geometry.

For fixed insertion/moderator settings, characterize each pattern's normalized fuel absorption, environment absorption, and fertility contribution as functions of fuel temperature. Incoming intensity supplies the scale. Combine the four source rays instead of interpreting their cells repeatedly.

Possible tick structure:

```text
fuel temperature
  -> radiation response
  -> four pattern responses
  -> combined heat/fertility contributions
  -> state update
```

**Experiment:** start with the default setting and account explicitly for intensity cutoffs. Compare direct precomputed expressions with tables. Avoid evaluating all 31 patterns each tick if that costs more than evaluating the four actually used.

### 6. Expand short ray sequences algebraically

Hardness is divided by known moderation factors along a fixed path. Absorption and remaining intensity can therefore be expanded into compact expressions, including small polynomials in initial hardness and temperature-dependent fuel response when guards/cutoffs are handled separately.

Precompute coefficients per source or pattern and evaluate the combined response rather than a branching ray interpreter.

**Potential:** preserve the equations while reducing repeated interpretation. **Uncertainty:** expression size, coefficient loads, register pressure, rounding, and polynomial-evaluation cost could outweigh savings. Benchmark both forms.

### 7. Tabulate complete temperature responses

Three nested exponential expressions use the same fuel temperature: radiation hardness, effective-radiation multiplier, and fuel absorption response.

Tabulate the final functions instead of approximating each exponential separately. Explore piecewise interpolation, nonuniform grids around high-curvature regions, and combination with the 31 ray-pattern responses.

**Experiment:** set an explicit error budget and domain; retain fallback or justified tail handling outside the table. Compare native math with realistic lookup/interpolation access patterns. A poorly cached table can be slower than native math. Ranking and near-floor behavior matter more than a checkerboard-only check.

### 8. Group by rod count and specialize counts

Candidates with the same rod count share fuel-volume constants and source-cycle length. Grouping them may improve branch coherence and topology memory access within GPU execution groups.

Explore rod-count-specific kernels for common search counts. Static counts may permit source-loop unrolling and eliminate dynamically indexed state/topology accesses. Precompute constants for counts 1 through width×depth (49 in the default footprint), keyed by height, fill, and insertion, instead of recomputing two powers per candidate.

**Caveat:** the powers occur once per candidate, whereas temperature exponentials occur every tick; optimizing those powers alone is unlikely to be a major gain. Grouping/specialization is the larger hypothesis.

### 9. Investigate per-thread topology storage

CUDA already specializes topology storage to width×depth×4 dynamically indexed ray bytes per thread (196 bytes for 7×7). It may cause local-memory traffic or register pressure; this has not been established with an instruction-level profile. Do not claim the default GPU kernel allocates the host's full 1,024-column topology.

Alternatives:

- Pack four rays for one source into one word. This does not inherently reduce the current 196-byte storage size, but can reduce loads and simplify extraction.
- Use a separate coalesced topology buffer.
- Use smaller storage in rod-count-specialized kernels.
- Unroll source accesses where the compiler can avoid dynamic arrays.
- Recompute inexpensive topology facts if loading them is costlier.

**Experiment:** inspect actual registers, spills/local-memory accesses, branch behavior, occupancy, and stall reasons before choosing a representation. Higher occupancy is not automatically faster.

### 10. Deduplicate and cache evaluations

Mutations around a small elite population can revisit the same mask many times. The baseline throughput count includes these repeated evaluations.

- Deduplicate a generated batch before simulation.
- Cache scores keyed by mask and all simulation settings/protocol/version.
- Replace duplicate proposals with new mutations.
- Cache topology independently of scores.

**Experiment:** first measure the duplicate rate and lookup overhead. Track cache hits separately from new simulations. The expected benefit is more distinct layouts explored per second, even if raw kernel throughput is unchanged.

### 11. Exploit symmetry and stronger simulation equivalence

Canonicalize rotations/reflections to avoid redundant layouts if the cost is worthwhile. Validate source-order effects: the row-major source sequence can slightly affect the simulated averages, so symmetry is not automatically bitwise equivalence in this numerical model.

Explore simulation fingerprints incorporating fuel count, heat-transfer coefficient, and the ordered sequence of source responses. Different masks with identical relevant behavior may share an evaluation.

**Experiment:** compare fingerprint-equivalent layouts against the reference. Distinguish mathematically equivalent behavior from approximate score equivalence; do not merge layouts solely because a coarse histogram happens to match.

### 12. Adaptive GPU convergence

GPU discovery currently runs every candidate for all 4,500 ticks. Stop candidates whose measured averages and relevant state behavior have stabilized.

- Align comparisons with source cycles.
- Use averaged power/fuel and phase-aware state comparisons.
- Avoid a single arbitrary temperature snapshot as the criterion.
- Process chunks and compact unfinished candidates to reduce slow-lane tail effects.

**Scale illustration:** reducing 4,500 ticks to 500 removes 89% of tick work. Whether 500 ticks is sufficient is unproven. Total throughput would improve by less than ninefold because host work, transfers, and other costs remain.

**Experiment:** use strict reference holdouts, difficult oscillatory layouts, and measurements of ranking/false early stopping. Keep full-budget fallbacks.

### 13. Warm-start candidates from parents or a state table

One-cell mutations often produce reactors similar to their parent. Inherit approximate temperatures/fertility, settle, and measure instead of starting at zero.

A more general initial-state table could be indexed by rod count, conductivity, and geometry features.

**Risk to investigate:** oscillation, multiple attractors, and source phase can make hot-start and cold-start results differ. Finalists still require cold-start verification. Sample ordinary candidates as well, because final verification alone does not reveal promising candidates mistakenly rejected by inaccurate discovery scores.

### 14. Solve steady or repeating behavior directly

The state is small: fuel temperature, reactor temperature, and fertility. Seek a fixed point of the complete source-cycle update or solve for a repeating trajectory using a shooting method, Anderson acceleration, or another nonlinear solver.

Some layouts may repeat only after several source cycles; others may not present a simple stable cycle. Retain ordinary simulation as a fallback.

**Experiment:** solve the actual discrete model, including guards/clamps. An average-source equilibrium model is a different approximation and must be labeled and tested as such.

### 15. Treat fertility as a separate subsystem

Under maintained fuel fill, fertility affects reported burn but does not drive thermal evolution. Above the minimum-decay threshold, its update is:

```text
F_next = 0.95 * F_current + 0.95 * absorbed_radiation
```

This suggests solving fertility from a repeating absorption sequence, computing thermal behavior first and fuel consumption afterward, or running only a bounded fertility preroll before measurement.

**Experiment:** derive a valid error bound and account for the minimum-decay/zero-clamp branches. This is more aggressive than idea 1: skipping warmup burn calculations preserves fertility updates, whereas shortening fertility evolution requires further proof or approximation validation.

### 16. Move evolution and selection to the GPU

Keep candidate masks, scores, mutation, crossover, and elite selection in device memory. Return only progress counters and champions. Consider independent device islands with occasional elite exchange.

Potential benefits include reduced output transfer, host selection, and synchronization. Finalists still go through CPU verification.

**Experiment:** measure current transfer/selection costs first. As simulation gets faster, these costs will become more important. Device RNG/state, selection kernels, register pressure, and changed search behavior must be included in end-to-end comparisons.

### 17. Pipeline work and reduce submission overhead

The current loop blocks through generate/upload/evaluate/synchronize/download/select. Double or triple buffering could overlap candidate preparation and asynchronous transfers with GPU evaluation. Pinned host buffers and streams are candidates.

CUDA Graphs may reduce repeated submission overhead, especially if adaptive convergence/selection introduces several short kernels. Graphs do not reduce physics work themselves.

**Experiment:** compare timelines and complete-search throughput, not just isolated API overhead. Specify how delayed elite updates in a pipeline affect the search.

### 18. Change GPU work assignment and launch configuration

Explore:

- Four lanes per reactor, each handling one ray of the same irradiation event.
- Small cooperative groups sharing state/topology.
- Different block sizes and register limits.
- Several independent candidates per thread to expose more independent work.

Potential benefits compete with extra synchronization, duplicated state, and resource use. The four-lane approach must not become four source irradiations per tick. Benchmark mapping variants on the actual RTX 3080 Ti; do not assume a full warp per candidate is automatically beneficial.

### 19. Progressive evaluation and early screening

Give all candidates a cheap first pass, promising candidates a longer pass, and contenders a strict verification.

Possible screens include short simulations, temperature estimates, or conservative power/efficiency bounds. Claimed safe bounds must cover the actual discrete model, including heat clamps; a simple physical energy-conservation argument may miss numerical clamp behavior.

Approximate screens may provide larger gains but need sampled checks for falsely rejected winners. Measure screening throughput separately from complete/validated simulation throughput.

### 20. Train a surrogate and explore tensor cores

Use unique evaluated layouts to train a small predictor of power and efficiency from masks and physical features: exposed fuel faces, rod count, conductivity, and ray-pattern statistics.

Screen large populations with the predictor, simulate promising or uncertain candidates, and periodically sample rejected candidates to identify blind spots. Keep cold-start reference verification for final results.

Tensor cores may be useful for the predictor or a suitable matrix formulation of response evaluation. They do not directly accelerate the current scalar, branching tick loop. Mixed precision should be explored selectively rather than assuming that all thermal state can safely become half precision.

## Additional ideas retained from the brainstorm

- **Incremental topology:** a flipped cell changes nearby heat-transfer faces and paths that encounter it within four steps. Update affected topology rather than rebuilding everything, while accounting for rod-count-dependent constants and source ordering.
- **Search quality:** independent islands, migration, configurable mutations and restarts are implemented. Future work: heterogeneous/adaptive strategies per agent, improvement-per-time budget allocation, novelty-aware elites, and a power/efficiency Pareto archive. Better verified results per wall-clock minute can matter more than a higher raw counter.
- **Concurrent CPU search:** explore independent CPU candidates while the GPU runs, provided this does not slow GPU feeding or contention-sensitive host work.
- **Multiple GPUs/machines:** independent search islands with occasional champion exchange; preserve settings/backend provenance and CPU verification.
- **Platform experiments:** compare Linux with the current Windows display-GPU setup using equivalent builds, masks, and settings. Outcome unknown.
- **Clock/thermal consistency:** record sustained clocks, temperature, and power behavior; benchmark with other GPU workloads idle. Do not confuse changed clocks with a code improvement.
- **Autotuning:** choose batch size, block size, specialization, and numerical mode based on measured behavior on the actual machine.

## Evaluation protocol for future experiments

### Keep metrics distinct

1. Full fixed-budget simulation evaluations/s on identical masks and settings.
2. Fast-screen/convergence evaluations/s, with their error and stopping protocol disclosed.
3. Distinct layouts explored/s; cache hits and duplicates separately.
4. Exact-verified candidate throughput and verification overhead.
5. Best verified FE/mB after a fixed search duration at a specified power floor.

### Proposed comparison procedure

- Record binary revision, hardware, compiler/options, GPU clocks, backend, workers, batch/block sizes, seed, and every simulation setting.
- Use identical fixed mask sets for microbenchmarks. Include sparse/dense/boundary layouts, near-tied candidates, near-floor cases, changed insertion/fill, and oscillatory cases.
- Warm up the GPU before timing; repeat enough to expose run-to-run variation.
- Time upload, kernel, download, generation, selection, and verification separately. Use GPU timing for the kernel rather than treating a host synchronization timer as pure kernel time.
- Inspect error distributions, ranking, top-K overlap, and power-floor classification. For screens/warm starts, inspect false rejection using sampled strict evaluations.
- Compare fixed-time searches over multiple seeds because changes to batch ordering, mutation, or elite updates alter the population being measured.
- Retain clear fallback modes and record which protocol produced a saved result.
- Preserve the existing baseline for A/B comparison. Do not claim an optimization helps merely because a single search produced a higher evaluations/s counter.

### Existing commands

Run from the project root, with the packaged executable and its NVRTC DLLs available:

```powershell
.\outputs\atm_er2_optimizer.exe --backend cuda --benchmark --threads 24 --seconds 10 --min-power 350000
.\outputs\atm_er2_optimizer.exe --backend cuda --self-test --validation-layouts 1024
python tests/regression.py --gpu
.\tools\profile.ps1 -Layouts 524288
```

Additional current developer checks after a CMake build:

```powershell
.\build-cmake\er2_performance.exe 65536 11
$env:ER2_GPU_PROFILE='1'
.\build-cmake\er2_performance.exe 65536 7
Remove-Item Env:ER2_GPU_PROFILE
.\build-cmake\gpu_accuracy_test.exe --double-stride 17
.\tools\local\cmake\data\bin\ctest.exe --test-dir build-cmake --output-on-failure
```

Transfer/kernel event timing and generation/reuse/selection counters are implemented. Nsight instruction profiling, lifetime unique-layout tracking, and predictor/convergence quality instrumentation remain future work.

## References checked during the brainstorm

- [NVIDIA CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html): arithmetic, memory access, transfers, occupancy, and divergence.
- [NVIDIA Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html): instruction mix, memory workload, occupancy, and stall analysis.
- [NVIDIA Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html): architecture-specific tuning context.
- [NVIDIA CUDA Graphs documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html): repeated workflow submission and graph execution.
- [ER2 1.21 ReactorLogic](https://github.com/ZeroNoRyouki/ExtremeReactors2/blob/1.21/src/main/java/it/zerono/mods/extremereactors/gamecontent/multiblock/reactor/ReactorLogic.java): model comparison context.

## Resume note

This pass is implemented and validated. Baseline binaries, raw CSVs, fixture results, PTX, and the rejected lookup prototype are in ignored `work/optimization`; summaries and reproduction instructions are in this file and PERFORMANCE.md. The full original brainstorm remains above. Recheck assumptions if fuel, geometry, source schedule, physics, compiler, or hardware changes. The pre-optimization baseline is commit `2222a5d`.

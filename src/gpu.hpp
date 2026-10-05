#pragma once
#include "reactor_core.hpp"
#include <memory>
#include <span>
#include <string>
namespace er2 {
struct GpuDiagnostics {
    int layout_words=0, input_bytes_per_layout=0, output_bytes_per_layout=0;
    int block_threads=128, multiprocessors=0, registers_per_thread=0;
    int local_bytes_per_thread=0, shared_bytes_per_block=0;
    bool double_precision=false, fma=false;
    bool packed_rays=false, unroll_rays=false, reciprocal_divisors=false;
    bool settings_specialized=false;
    bool group_rods=false, branchless_rays=false;
};
struct GpuTimings {
    double packing_ms=0, upload_ms=0, kernel_ms=0, download_ms=0, unpacking_ms=0, total_ms=0;
    int layouts=0;
    bool grouping_applied=false;
    int distinct_rod_counts=0;
};
class Gpu {
public:
    Gpu(bool exact, int device, int capacity, int cells=49,const Settings* specialized_settings=nullptr);
    ~Gpu();
    Gpu(const Gpu&)=delete;
    Gpu& operator=(const Gpu&)=delete;
    void evaluate(std::span<const Layout> masks,std::span<Result> output,Settings settings,SimConfig config);
    const std::string& name() const;
    const GpuDiagnostics& diagnostics() const;
    const GpuTimings& timings() const;
    // CUDA event collection is optional so search batches avoid profiling overhead.
    void set_profiling(bool enabled);
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
}

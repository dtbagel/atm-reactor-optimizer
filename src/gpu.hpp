#pragma once
#include "reactor_core.hpp"
#include <memory>
#include <span>
#include <string>
namespace er2 {
class Gpu {
public:
    Gpu(bool exact, int device, int capacity, int cells=49);
    ~Gpu();
    Gpu(const Gpu&)=delete;
    Gpu& operator=(const Gpu&)=delete;
    void evaluate(std::span<const Layout> masks,std::span<Result> output,Settings settings,SimConfig config);
    const std::string& name() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
}

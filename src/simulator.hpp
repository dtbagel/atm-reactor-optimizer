#pragma once
#include "reactor_core.hpp"
namespace er2 {
Result evaluate_exact(Layout mask, Settings settings, SimConfig config, bool adaptive=true);
constexpr Layout checkerboard() { return 0x1555555555555ULL; }
}

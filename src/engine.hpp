#pragma once
#include "reactor_core.hpp"
#include <atomic>
#include <functional>
#include <mutex>
#include <string>

namespace er2 {
struct SearchTuning {
    int agents=4, power_agents=1, elites=64;
    int mutation_percent=90, max_flips=5, move_percent=45;
    int crossover_percent=20, random_percent=20;
    int migration_generations=25, restart_generations=100;
};
inline bool valid_tuning(const SearchTuning& t) {
    return t.agents>=1&&t.agents<=32&&t.power_agents>=0&&t.power_agents<=t.agents&&t.elites>=1&&t.elites<=256&&t.max_flips>=1&&t.max_flips<=1024&&
        t.mutation_percent>=0&&t.mutation_percent<=100&&t.move_percent>=0&&t.move_percent<=100&&t.crossover_percent>=0&&t.crossover_percent<=100&&t.random_percent>=0&&t.random_percent<=100&&
        t.migration_generations>=0&&t.migration_generations<=1000000&&t.restart_generations>=0&&t.restart_generations<=1000000;
}
struct Progress {
    double seconds=0, budget=0;
    unsigned long long candidates=0, ticks=0;
    Result best;
    bool feasible=false;
    unsigned long long generation=0, improvements=0, restarts=0, since_improvement=0;
    double last_improvement_seconds=0;
    int agents=1, distinct_parents=0;
    // Candidates count proposals; only simulations consume reactor ticks.
    unsigned long long simulation_evaluations=0, cache_hits=0, batch_duplicates=0;
    double generation_seconds=0, simulation_seconds=0, reuse_seconds=0, selection_seconds=0;
    double discovery_minimum_power=0;
};
// Callbacks run on the optimizer worker, never on the UI/render thread.
struct RunControl {
    std::atomic<bool> finish{false};
    std::atomic<bool> reseed{false};
    void set_tuning(SearchTuning tuning) {if(valid_tuning(tuning)){std::lock_guard lock(tuning_mutex);live_tuning=tuning;has_tuning=true;}}
    SearchTuning get_tuning(SearchTuning fallback) {std::lock_guard lock(tuning_mutex);return has_tuning?live_tuning:fallback;}
    std::function<void(const std::string&)> log;
    std::function<void(const std::string&)> stage;
    std::function<void(const std::string&)> phase;
    std::function<void(const Progress&)> progress;
    std::function<void(const Result&)> verified;
private:
    std::mutex tuning_mutex;
    SearchTuning live_tuning;
    bool has_tuning=false;
};
int optimizer_main(int argc,char** argv,RunControl* control=nullptr);
std::string mask_hex(Layout mask);
std::string layout_text(Layout mask,Settings settings);
}

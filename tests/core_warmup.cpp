#include "moderators.hpp"
#include <bit>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <type_traits>

template<class T> bool same(T a, T b) {
    using Bits = std::conditional_t<sizeof(T)==4,std::uint32_t,std::uint64_t>;
    return std::bit_cast<Bits>(a)==std::bit_cast<Bits>(b);
}

template<class T> void check(er2::Layout mask, er2::Settings settings, er2::SimConfig config) {
    er2::Topology topology(mask,settings);
    if(er2::rod_count(mask)!=topology.rods) throw std::runtime_error("Native popcount changed the fuel-column count");
    er2::State<T> observed(topology,settings), warmup(topology,settings), branchless(topology,settings);
    const int samples=er2::minimum(config.sample_ticks,config.max_ticks);
    T sum_power=0,sum_fuel=0;
    int source=0;
    for (int tick=0;tick<config.max_ticks;++tick) {
        T power,fuel,discarded_power,discarded_fuel,branchless_power,branchless_fuel;
        observed.template step<true,false>(topology,source,power,fuel);
        warmup.template step<false,false>(topology,source,discarded_power,discarded_fuel);
        branchless.template step<true,true>(topology,source,branchless_power,branchless_fuel);
        if (!same(observed.fuel_heat,warmup.fuel_heat) ||
            !same(observed.reactor_heat,warmup.reactor_heat) ||
            !same(observed.fertility,warmup.fertility))
            throw std::runtime_error("Warmup changed a simulation state bit");
        if (!same(observed.fuel_heat,branchless.fuel_heat) ||
            !same(observed.reactor_heat,branchless.reactor_heat) ||
            !same(observed.fertility,branchless.fertility) ||
            !same(power,branchless_power) || !same(fuel,branchless_fuel))
            throw std::runtime_error("Branchless recurrence changed a simulation bit");
        if (tick>=config.max_ticks-samples) {sum_power+=power;sum_fuel+=fuel;}
        if (++source==topology.rods) source=0;
    }
    auto before=observed.result(mask,topology.rods,config.max_ticks,sum_power/T(samples),sum_fuel/T(samples));
    auto after=er2::evaluate_fixed<T>(mask,settings,config);
    if (!same(before.power,after.power) || !same(before.fuel,after.fuel) ||
        !same(before.efficiency,after.efficiency) || !same(before.fuel_heat,after.fuel_heat) ||
        !same(before.reactor_heat,after.reactor_heat) || !same(before.fertility,after.fertility))
        throw std::runtime_error("Split fixed evaluator changed a result bit");
}

int main() {
    try {
        constexpr int shapes[][3]={{7,7,7},{3,5,4},{1,9,6},{9,1,8},{9,8,5},{16,12,32},{32,32,64},{1,1,1}};
        int count=0;
        for (const auto& preset:er2::moderators) {
            const auto& shape=shapes[count%8];
            er2::Settings settings;
            settings.width=shape[0];settings.depth=shape[1];settings.height=shape[2];settings.moderator=preset.material;
            settings.insertion=(count%4)*0.25;settings.fill=(count%3+1)/3.0;settings.variant=(count%2)?0.5:1;
            er2::Layout mask;
            for(int cell=0;cell<settings.cells();++cell) if(cell%17==count%17) mask.set(cell);
            if(!mask) mask.set(settings.cells()-1);
            er2::SimConfig config{count%13==0?4500:513,1,count%3==0?1:count%3==1?113:8192};
            check<double>(mask,settings,config);check<float>(mask,settings,config);
            ++count;
        }
        // Dense topology, zero fuel fill, total insertion, and source-cycle tails.
        for(int boundary=0;boundary<3;++boundary) {
            er2::Settings settings;
            if(boundary==0) settings.insertion=1;
            if(boundary==1) settings.fill=0;
            const auto mask=er2::full_mask(settings.cells());
            check<double>(mask,settings,{4500,1,500});check<float>(mask,settings,{4500,1,500});
            ++count;
        }
        std::cout << "Warmup, branchless rays, and split-window results match every bit in " << count << " float/double cases\n";
    } catch (const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}

#include "simulator.hpp"
#include <array>
#include <algorithm>
#include <stdexcept>
namespace er2 {
Result evaluate_exact(Layout mask, Settings settings, SimConfig config, bool adaptive) {
    if (!adaptive) return evaluate_fixed<double>(mask,settings,config);
    if (config.sample_ticks<1 || config.sample_ticks>8192) throw std::runtime_error("sample ticks must be 1..8192");
    Topology top(mask);
    if (!top.rods) return {};
    State<double> state(top,settings);
    std::array<double,8192> powers{}, fuels{};
    int count=0,index=0,source=0,stable=0,ticks=0;
    bool have_snapshot=false;
    double old_fuel=0,old_reactor=0,old_fertility=0;
    for (int tick=0;tick<config.max_ticks;++tick) {
        double power,fuel;
        state.step(top,source,power,fuel);
        if (++source==top.rods) source=0;
        powers[index]=power; fuels[index]=fuel;
        index=(index+1)%config.sample_ticks;
        count=std::min(config.sample_ticks,count+1);
        ticks=tick+1;
        if (ticks>=config.min_ticks && ticks%250==0) {
            if (have_snapshot) {
                double delta=std::max({std::abs(state.fuel_heat-old_fuel)/std::max(1.,std::abs(state.fuel_heat)),
                    std::abs(state.reactor_heat-old_reactor)/std::max(1.,std::abs(state.reactor_heat)),
                    std::abs(state.fertility-old_fertility)/std::max(1.,std::abs(state.fertility))});
                stable=delta<2e-5 ? stable+1 : 0;
                if (stable>=2 && count>=config.sample_ticks) break;
            }
            old_fuel=state.fuel_heat; old_reactor=state.reactor_heat; old_fertility=state.fertility;
            have_snapshot=true;
        }
    }
    double power=0,fuel=0;
    for (int i=0;i<count;++i) { power+=powers[i]; fuel+=fuels[i]; }
    return state.result(mask,top.rods,ticks,power/count,fuel/count);
}
}

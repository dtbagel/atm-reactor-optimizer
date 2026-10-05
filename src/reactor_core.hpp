#pragma once
// Shared equations for the double CPU reference and CUDA kernels.
// No runtime allocations, object model, or per-tick topology discovery.
#ifndef __CUDACC_RTC__
#include <bit>
#include <cmath>
#define ER2_INLINE inline
#else
#define ER2_INLINE __device__ __forceinline__
#endif

namespace er2 {
// Full host footprint; device compilation can specialize the wire representation.
#ifndef ER2_LAYOUT_WORDS
#define ER2_LAYOUT_WORDS 16
#endif
struct Layout {
    unsigned long long words[ER2_LAYOUT_WORDS]{};
    ER2_INLINE constexpr Layout(unsigned long long value=0) { words[0]=value; }
    ER2_INLINE bool test(int cell) const { return (words[cell/64] >> (cell%64)) & 1ULL; }
    ER2_INLINE void toggle(int cell) { words[cell/64] ^= 1ULL << (cell%64); }
    ER2_INLINE void set(int cell) { words[cell/64] |= 1ULL << (cell%64); }
    ER2_INLINE explicit operator bool() const { for (auto w:words) if(w) return true; return false; }
    ER2_INLINE bool operator==(const Layout& b) const { for(int i=0;i<ER2_LAYOUT_WORDS;++i) if(words[i]!=b.words[i]) return false; return true; }
    ER2_INLINE bool operator!=(const Layout& b) const { return !(*this==b); }
    ER2_INLINE bool operator<(const Layout& b) const { for(int i=ER2_LAYOUT_WORDS-1;i>=0;--i) if(words[i]!=b.words[i]) return words[i]<b.words[i]; return false; }
    ER2_INLINE Layout operator&(const Layout& b) const { Layout r; for(int i=0;i<ER2_LAYOUT_WORDS;++i) r.words[i]=words[i]&b.words[i]; return r; }
    ER2_INLINE Layout operator|(const Layout& b) const { Layout r; for(int i=0;i<ER2_LAYOUT_WORDS;++i) r.words[i]=words[i]|b.words[i]; return r; }
    ER2_INLINE Layout operator~() const { Layout r; for(int i=0;i<ER2_LAYOUT_WORDS;++i) r.words[i]=~words[i]; return r; }
};
ER2_INLINE int rod_count(const Layout& mask) {
    int n=0;
    for(auto w:mask.words) {
#ifdef __CUDACC_RTC__
        n+=__popcll(w);
#else
        n+=std::popcount(w);
#endif
    }
    return n;
}
ER2_INLINE Layout full_mask(int cells) { Layout m; for(int i=0;i<cells;++i) m.set(i); return m; }
constexpr Layout all_mask = (1ULL << 49) - 1;
struct Moderator { double absorption=0.972, heat_efficiency=0.91, moderation=3.074, conductivity=5; };
struct Settings {
    double insertion=0, fill=1, variant=1;
    Moderator moderator;
    int width=7, depth=7, height=7;
    ER2_INLINE int cells() const { return width*depth; }
};
struct SimConfig { int max_ticks = 4500, min_ticks = 1500, sample_ticks = 500; };
struct Result {
    Layout mask = 0;
    double power = 0, fuel = 0, efficiency = 0;
    double fuel_heat = 0, reactor_heat = 0, fertility = 0;
    int rods = 0, ticks = 0;
};
template<class T> ER2_INLINE T maximum(T a, T b) { return a > b ? a : b; }
template<class T> ER2_INLINE T minimum(T a, T b) { return a < b ? a : b; }
#ifdef ER2_BRANCHLESS_RAYS
inline constexpr bool default_branchless_rays = true;
#else
inline constexpr bool default_branchless_rays = false;
#endif
template<class T> struct Math;
template<> struct Math<double> {
    ER2_INLINE static double exponential(double x) { return ::exp(x); }
    ER2_INLINE static double power(double x, double y) { return ::pow(x, y); }
    ER2_INLINE static double logarithm(double x) { return ::log10(x); }
};
template<> struct Math<float> {
    ER2_INLINE static float exponential(float x) {
#ifdef __CUDACC_RTC__
        return __expf(x);
#else
        return ::expf(x);
#endif
    }
    ER2_INLINE static float power(float x, float y) {
#ifdef __CUDACC_RTC__
        return __powf(x, y);
#else
        return ::powf(x, y);
#endif
    }
    ER2_INLINE static float logarithm(float x) {
#ifdef __CUDACC_RTC__
        return __log10f(x);
#else
        return ::log10f(x);
#endif
    }
};
struct Topology {
    // low three bits: ray length; high four bits: fuel at each step.
#ifndef ER2_TOPOLOGY_CELLS
#define ER2_TOPOLOGY_CELLS 1024
#endif
#ifdef ER2_PACKED_RAYS
    unsigned int rays[ER2_TOPOLOGY_CELLS];
#else
    unsigned char rays[ER2_TOPOLOGY_CELLS][4];
#endif
    int rods = 0;
    double heat_transfer = 0;
    ER2_INLINE explicit Topology(Layout mask, Settings settings={}) {
        const int dx[4] = {1,-1,0,0}, dz[4] = {0,0,1,-1};
        for (int cell = 0; cell < settings.cells(); ++cell) {
            if (!mask.test(cell)) continue;
            int x = cell % settings.width, z = cell / settings.width;
#ifdef ER2_PACKED_RAYS
            rays[rods] = 0;
#endif
            for (int d = 0; d < 4; ++d) {
                int nx = x + dx[d], nz = z + dz[d];
                if (nx < 0 || nx >= settings.width || nz < 0 || nz >= settings.depth) heat_transfer += 0.6;
                else if (!mask.test(nz*settings.width+nx)) heat_transfer += settings.moderator.conductivity;
                int length = 0, fuel_bits = 0;
                for (int step = 0; step < 4; ++step) {
                    if (nx < 0 || nx >= settings.width || nz < 0 || nz >= settings.depth) break;
                    if (mask.test(nz*settings.width+nx)) fuel_bits |= 1 << step;
                    ++length;
                    nx += dx[d]; nz += dz[d];
                }
                const unsigned int ray = static_cast<unsigned char>(length | (fuel_bits << 3));
#ifdef ER2_PACKED_RAYS
                rays[rods] |= ray << (8*d);
#else
                rays[rods][d] = static_cast<unsigned char>(ray);
#endif
            }
            ++rods;
        }
        heat_transfer *= settings.height;
    }
};
template<class T> struct State {
    T fuel_heat = 0, reactor_heat = 0, fertility = 1;
    T raw, scaled, moderation, insertion, variant, fuel_capacity, htc;
    T absorption, heat_efficiency, moderator_moderation, reactor_capacity, cooling, heat_loss;
#ifdef ER2_RECIPROCAL_DIVISORS
    T inverse_fuel_capacity, inverse_reactor_capacity, inverse_moderation, inverse_moderator_moderation;
#endif
    ER2_INLINE State(const Topology& top, Settings settings) {
        insertion = T(settings.insertion); variant = T(settings.variant);
        fuel_capacity = T(top.rods * settings.height * 10);
        htc = T(top.heat_transfer);
        absorption=T(settings.moderator.absorption); heat_efficiency=T(settings.moderator.heat_efficiency);
        moderator_moderation=T(settings.moderator.moderation);
        reactor_capacity=T(settings.width*settings.depth*settings.height*10);
        cooling=T(0.6)*T(2*(settings.width*settings.depth+settings.width*settings.height+settings.depth*settings.height));
        heat_loss=T(0.001)*T(2*((settings.width+2)*(settings.depth+2)+(settings.width+2)*(settings.height+2)+(settings.depth+2)*(settings.height+2)));
        T modifier = T(1) - insertion;
        raw = T(top.rods * settings.height * 4000) * T(settings.fill) * T(0.01) * modifier;
        scaled = Math<T>::power(raw, T(1.05));
        scaled = Math<T>::power(scaled / T(top.rods), T(1.05)) * T(top.rods) * modifier;
        moderation = T(1.5) + T(1.5)*insertion + insertion;
#ifdef ER2_RECIPROCAL_DIVISORS
        inverse_fuel_capacity = T(1)/fuel_capacity;
        inverse_reactor_capacity = T(1)/reactor_capacity;
        inverse_moderation = T(1)/moderation;
        inverse_moderator_moderation = T(1)/moderator_moderation;
#endif
    }
    template<bool Observe = true, bool Branchless = default_branchless_rays>
    ER2_INLINE void step(const Topology& top, int source, T& power, T& fuel_usage) {
        T hardness0 = T(0.2) + T(0.8)*Math<T>::exponential(T(-15)*Math<T>::exponential(T(-0.0025)*fuel_heat));
        // Maintained fuel fill makes consumption an observation only: fertility
        // still evolves on every tick, but warmup need not calculate its logarithm.
        if constexpr (Observe) {
            T fertility_modifier = fertility <= T(1) ? T(1) : Math<T>::logarithm(fertility) + T(1);
            fuel_usage = T(0.0007)*raw / fertility_modifier * T(0.8);
        }
        T effective = scaled*(T(1)-T(0.95)*Math<T>::exponential(T(-10)*Math<T>::exponential(T(-0.0012)*fuel_heat)));
        // This response is shared by every fuel encounter this tick.
        T response = T(1)-T(0.95)*Math<T>::exponential(T(-10)*Math<T>::exponential(T(-0.0022)*fuel_heat));
        T fuel_energy = T(10)*effective, environment_energy = 0, absorbed_fuel = 0;
#ifdef ER2_PACKED_RAYS
        const unsigned int source_rays = top.rays[source];
#endif
#if defined(__CUDACC_RTC__) && defined(ER2_UNROLL_RAYS)
#pragma unroll 4
#endif
        for (int direction = 0; direction < 4; ++direction) {
#ifdef ER2_PACKED_RAYS
            unsigned int ray = (source_rays >> (8*direction)) & 255;
#else
            unsigned int ray = top.rays[source][direction];
#endif
            int length = ray & 7;
            unsigned int fuel_bits = ray >> 3;
            T intensity = effective*T(0.25), hardness = hardness0;
#if defined(__CUDACC_RTC__) && (defined(ER2_UNROLL_RAYS) || defined(ER2_BRANCHLESS_RAYS))
#pragma unroll 4
#endif
            if constexpr (Branchless) {
            for (int step = 0; step < 4; ++step) {
                if (step>=length || intensity <= T(0.0001)) break;
                // Evaluate both material responses, then select with 0/1 weights.
                // Length and intensity guards retain the generic ray semantics.
                const T is_fuel = T((fuel_bits >> step) & 1U);
                const T is_moderator = T(1)-is_fuel;
                const T fuel_absorption = minimum(T(1),response*(T(1)-hardness)*T(0.5));
                const T bonus = (T(1)-fuel_absorption)*insertion*T(0.5);
                const T penalty = fuel_absorption*insertion*T(0.5);
                const T absorbed_by_fuel = (fuel_absorption+bonus)*intensity;
                const T absorbed_by_moderator = intensity*absorption*(T(1)-hardness);
                absorbed_fuel += is_fuel*((fuel_absorption-penalty)*intensity);
                intensity = maximum(T(0),intensity-(is_fuel*absorbed_by_fuel+is_moderator*absorbed_by_moderator));
#ifdef ER2_RECIPROCAL_DIVISORS
                hardness *= is_fuel*inverse_moderation+is_moderator*inverse_moderator_moderation;
#else
                hardness /= is_fuel*moderation+is_moderator*moderator_moderation;
#endif
                fuel_energy += is_fuel*(absorbed_by_fuel*T(10));
                environment_energy += is_moderator*(heat_efficiency*absorbed_by_moderator*T(10));
            }
            } else {
            for (int step = 0; step < length; ++step) {
                if (intensity <= T(0.0001)) break;
                if (fuel_bits & (1U << step)) {
                    T absorption = minimum(T(1), response*(T(1)-hardness)*T(0.5));
                    T bonus = (T(1)-absorption)*insertion*T(0.5);
                    T penalty = absorption*insertion*T(0.5);
                    T absorbed = (absorption+bonus)*intensity;
                    absorbed_fuel += (absorption-penalty)*intensity;
                    intensity = maximum(T(0), intensity-absorbed);
#ifdef ER2_RECIPROCAL_DIVISORS
                    hardness *= inverse_moderation;
#else
                    hardness /= moderation;
#endif
                    fuel_energy += absorbed*T(10);
                } else {
                    T absorbed = intensity*absorption*(T(1)-hardness);
                    intensity = maximum(T(0), intensity-absorbed);
#ifdef ER2_RECIPROCAL_DIVISORS
                    hardness *= inverse_moderator_moderation;
#else
                    hardness /= moderator_moderation;
#endif
                    environment_energy += heat_efficiency*absorbed*T(10);
                }
            }
            }
        }
        fertility += absorbed_fuel;
#ifdef ER2_RECIPROCAL_DIVISORS
        fuel_heat += fuel_energy*inverse_fuel_capacity;
        reactor_heat += environment_energy*inverse_reactor_capacity;
#else
        fuel_heat += fuel_energy/fuel_capacity;
        reactor_heat += environment_energy/reactor_capacity;
#endif
        fertility = maximum(T(0), fertility-maximum(T(0.1),fertility/T(20)));
        T difference = fuel_heat-reactor_heat;
        if (difference > T(0.01)) {
            T transferred = difference*htc;
#ifdef ER2_RECIPROCAL_DIVISORS
            fuel_heat = (fuel_heat*fuel_capacity-transferred)*inverse_fuel_capacity;
            reactor_heat = (reactor_heat*reactor_capacity+transferred)*inverse_reactor_capacity;
#else
            fuel_heat = (fuel_heat*fuel_capacity-transferred)/fuel_capacity;
            reactor_heat = (reactor_heat*reactor_capacity+transferred)/reactor_capacity;
#endif
        }
        difference = reactor_heat-T(20);
        power = 0;
        if (difference > T(0.01)) {
            T transferred = difference*cooling;
            T energy = reactor_heat*reactor_capacity;
            transferred *= T(0.2);
            if constexpr (Observe) power = transferred*T(0.5)*T(12)*variant;
#ifdef ER2_RECIPROCAL_DIVISORS
            reactor_heat = (energy-transferred)*inverse_reactor_capacity;
#else
            reactor_heat = (energy-transferred)/reactor_capacity;
#endif
        }
        difference = reactor_heat-T(20);
        if (difference > T(0.000001)) {
            T lost = maximum(T(1),difference*heat_loss);
#ifdef ER2_RECIPROCAL_DIVISORS
            reactor_heat = maximum(T(0),reactor_heat*reactor_capacity-lost)*inverse_reactor_capacity;
#else
            reactor_heat = maximum(T(0),reactor_heat*reactor_capacity-lost)/reactor_capacity;
#endif
        }
        fuel_heat = maximum(T(0),fuel_heat);
        reactor_heat = maximum(T(0),reactor_heat);
    }
    ER2_INLINE Result result(Layout mask, int rods, int ticks, T power, T fuel) const {
        Result out;
        out.mask=mask; out.rods=rods; out.ticks=ticks;
        out.power=double(power); out.fuel=double(fuel);
        out.efficiency=fuel>T(0) ? double(power/fuel) : 0;
        out.fuel_heat=double(fuel_heat); out.reactor_heat=double(reactor_heat); out.fertility=double(fertility);
        return out;
    }
};
// Bounded fixed-tick evaluator: just accumulators, no GPU sample-window spills.
template<class T> ER2_INLINE Result evaluate_fixed(Layout mask, Settings settings, SimConfig config) {
    Topology top(mask,settings);
    if (!top.rods) return Result{};
    State<T> state(top, settings);
    T sum_power=0, sum_fuel=0;
    int sample = minimum(config.sample_ticks,config.max_ticks), source = 0;
    const int warmup = config.max_ticks-sample;
    for (int tick=0; tick<warmup; ++tick) {
        T power, fuel;
        state.template step<false>(top,source,power,fuel);
        if (++source==top.rods) source=0;
    }
    for (int tick=warmup; tick<config.max_ticks; ++tick) {
        T power, fuel;
        state.step(top,source,power,fuel);
        if (++source==top.rods) source=0;
        sum_power+=power; sum_fuel+=fuel;
    }
    return state.result(mask,top.rods,config.max_ticks,sum_power/T(sample),sum_fuel/T(sample));
}
} // namespace er2

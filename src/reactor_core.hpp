#pragma once
// Shared equations for the double CPU reference and CUDA kernels.
// No runtime allocations, object model, or per-tick topology discovery.
#ifndef __CUDACC_RTC__
#include <cmath>
#define ER2_INLINE inline
#else
#define ER2_INLINE __device__ __forceinline__
#endif

namespace er2 {
using Layout = unsigned long long;
constexpr Layout all_mask = (1ULL << 49) - 1;
struct Settings { double insertion = 0, fill = 1, variant = 1; };
struct SimConfig { int max_ticks = 4500, min_ticks = 1500, sample_ticks = 500; };
struct Result {
    Layout mask = 0;
    double power = 0, fuel = 0, efficiency = 0;
    double fuel_heat = 0, reactor_heat = 0, fertility = 0;
    int rods = 0, ticks = 0;
};
template<class T> ER2_INLINE T maximum(T a, T b) { return a > b ? a : b; }
template<class T> ER2_INLINE T minimum(T a, T b) { return a < b ? a : b; }
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
    unsigned char rays[49][4];
    int rods = 0;
    double heat_transfer = 0;
    ER2_INLINE explicit Topology(Layout mask) {
        const int dx[4] = {1,-1,0,0}, dz[4] = {0,0,1,-1};
        for (int cell = 0; cell < 49; ++cell) {
            if (!(mask & (1ULL << cell))) continue;
            int x = cell % 7, z = cell / 7;
            for (int d = 0; d < 4; ++d) {
                int nx = x + dx[d], nz = z + dz[d];
                if (nx < 0 || nx >= 7 || nz < 0 || nz >= 7) heat_transfer += 0.6;
                else if (!(mask & (1ULL << (nz*7+nx)))) heat_transfer += 5.0;
                int length = 0, fuel_bits = 0;
                for (int step = 0; step < 4; ++step) {
                    if (nx < 0 || nx >= 7 || nz < 0 || nz >= 7) break;
                    if (mask & (1ULL << (nz*7+nx))) fuel_bits |= 1 << step;
                    ++length;
                    nx += dx[d]; nz += dz[d];
                }
                rays[rods][d] = static_cast<unsigned char>(length | (fuel_bits << 3));
            }
            ++rods;
        }
        heat_transfer *= 7;
    }
};
template<class T> struct State {
    T fuel_heat = 0, reactor_heat = 0, fertility = 1;
    T raw, scaled, moderation, insertion, variant, fuel_capacity, htc;
    ER2_INLINE State(const Topology& top, Settings settings) {
        insertion = T(settings.insertion); variant = T(settings.variant);
        fuel_capacity = T(top.rods * 7 * 10);
        htc = T(top.heat_transfer);
        T modifier = T(1) - insertion;
        raw = T(top.rods * 7 * 4000) * T(settings.fill) * T(0.01) * modifier;
        scaled = Math<T>::power(raw, T(1.05));
        scaled = Math<T>::power(scaled / T(top.rods), T(1.05)) * T(top.rods) * modifier;
        moderation = T(1.5) + T(1.5)*insertion + insertion;
    }
    ER2_INLINE void step(const Topology& top, int source, T& power, T& fuel_usage) {
        T hardness0 = T(0.2) + T(0.8)*Math<T>::exponential(T(-15)*Math<T>::exponential(T(-0.0025)*fuel_heat));
        T fertility_modifier = fertility <= T(1) ? T(1) : Math<T>::logarithm(fertility) + T(1);
        fuel_usage = T(0.0007)*raw / fertility_modifier * T(0.8);
        T effective = scaled*(T(1)-T(0.95)*Math<T>::exponential(T(-10)*Math<T>::exponential(T(-0.0012)*fuel_heat)));
        // This response is shared by every fuel encounter this tick.
        T response = T(1)-T(0.95)*Math<T>::exponential(T(-10)*Math<T>::exponential(T(-0.0022)*fuel_heat));
        T fuel_energy = T(10)*effective, environment_energy = 0, absorbed_fuel = 0;
        for (int direction = 0; direction < 4; ++direction) {
            unsigned int ray = top.rays[source][direction];
            int length = ray & 7;
            unsigned int fuel_bits = ray >> 3;
            T intensity = effective*T(0.25), hardness = hardness0;
            for (int step = 0; step < length; ++step) {
                if (intensity <= T(0.0001)) break;
                if (fuel_bits & (1U << step)) {
                    T absorption = minimum(T(1), response*(T(1)-hardness)*T(0.5));
                    T bonus = (T(1)-absorption)*insertion*T(0.5);
                    T penalty = absorption*insertion*T(0.5);
                    T absorbed = (absorption+bonus)*intensity;
                    absorbed_fuel += (absorption-penalty)*intensity;
                    intensity = maximum(T(0), intensity-absorbed);
                    hardness /= moderation;
                    fuel_energy += absorbed*T(10);
                } else {
                    T absorbed = intensity*T(0.972)*(T(1)-hardness);
                    intensity = maximum(T(0), intensity-absorbed);
                    hardness /= T(3.074);
                    environment_energy += T(0.91)*absorbed*T(10);
                }
            }
        }
        fertility += absorbed_fuel;
        fuel_heat += fuel_energy/fuel_capacity;
        reactor_heat += environment_energy/T(3430);
        fertility = maximum(T(0), fertility-maximum(T(0.1),fertility/T(20)));
        T difference = fuel_heat-reactor_heat;
        if (difference > T(0.01)) {
            T transferred = difference*htc;
            fuel_heat = (fuel_heat*fuel_capacity-transferred)/fuel_capacity;
            reactor_heat = (reactor_heat*T(3430)+transferred)/T(3430);
        }
        difference = reactor_heat-T(20);
        power = 0;
        if (difference > T(0.01)) {
            T transferred = difference*T(176.4);
            T energy = reactor_heat*T(3430);
            transferred *= T(0.2);
            power = transferred*T(0.5)*T(12)*variant;
            reactor_heat = (energy-transferred)/T(3430);
        }
        difference = reactor_heat-T(20);
        if (difference > T(0.000001)) {
            T lost = maximum(T(1),difference*T(0.486));
            reactor_heat = maximum(T(0),reactor_heat*T(3430)-lost)/T(3430);
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
    Topology top(mask);
    if (!top.rods) return Result{};
    State<T> state(top, settings);
    T sum_power=0, sum_fuel=0;
    int sample = minimum(config.sample_ticks,config.max_ticks), source = 0;
    for (int tick=0; tick<config.max_ticks; ++tick) {
        T power, fuel;
        state.step(top,source,power,fuel);
        if (++source==top.rods) source=0;
        if (tick>=config.max_ticks-sample) { sum_power+=power; sum_fuel+=fuel; }
    }
    return state.result(mask,top.rods,config.max_ticks,sum_power/T(sample),sum_fuel/T(sample));
}
} // namespace er2

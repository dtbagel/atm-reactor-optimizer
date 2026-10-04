#!/usr/bin/env python3
"""
ATM10 Extreme Reactors 2 layout optimizer
==========================================

Optimizes a 9x9x9 reinforced passive reactor (7x7x7 interior) using:
  - Uranium/Yellorium fuel
  - Pure Unobtainium in every non-fuel interior column
  - ATM10 multipliers: 0.8x fuel, 4x global power, 3x reactor power
  - Extreme Reactors 2's one-fuel-rod-per-tick irradiation model

The search uses multiple PROCESSES (the --threads option) because Python threads
would be limited by the GIL for this CPU-heavy workload.

Default objective: maximize FE per mB of fuel.
Use --min-power if you want the most efficient layout above a power target.

Example:
    python er2_atm10_optimizer.py --threads 16 --seconds 60
    python er2_atm10_optimizer.py --threads 16 --seconds 60 --min-power 500000

Legend in output:
    R = fuel rod column (7 fuel rod blocks tall)
    U = Unobtainium moderator column (7 blocks tall)
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# Reactor geometry
# ---------------------------------------------------------------------------

W = 7
D = 7
H = 7
CELL_COUNT = W * D
REACTOR_VOLUME = W * D * H

DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))

# Interior surface area (7x7x7) and exterior surface area (9x9x9)
INNER_SURFACE_AREA = 2 * (W * D + W * H + D * H)
OUTER_SURFACE_AREA = 2 * ((W + 2) * (D + 2) + (W + 2) * (H + 2) + (D + 2) * (H + 2))

# ---------------------------------------------------------------------------
# Extreme Reactors 2 constants / fuel properties
# ---------------------------------------------------------------------------

AMBIENT_TEMP = 20.0
ENERGY_PER_RADIATION_UNIT = 10.0
ENERGY_PER_C_PER_UNIT_VOLUME = 10.0

PASSIVE_COOLING_TRANSFER_EFFICIENCY = 0.20
PASSIVE_COOLING_POWER_EFFICIENCY = 0.50
REACTOR_TO_COOLANT_COEFF = 0.60 * INNER_SURFACE_AREA
REACTOR_HEAT_LOSS_COEFF = 0.001 * OUTER_SURFACE_AREA

# Standard Yellorium/Uranium fuel
FUEL_MODERATION_FACTOR = 1.5
FUEL_ABSORPTION_COEFF = 0.5
FUEL_HARDNESS_DIVISOR = 1.0
FISSION_EVENTS_PER_FUEL_UNIT = 0.01
FUEL_UNITS_PER_FISSION_EVENT = 0.0007
FUEL_REACTIVITY = 1.05

# One fuel rod block holds 4 B = 4000 mB of reactant.
FUEL_CAPACITY_PER_ROD_BLOCK = 4000.0

# ---------------------------------------------------------------------------
# ATM10 configuration
# ---------------------------------------------------------------------------

ATM10_FUEL_USAGE_MULTIPLIER = 0.8
ATM10_POWER_PRODUCTION_MULTIPLIER = 4.0
ATM10_REACTOR_POWER_MULTIPLIER = 3.0
ATM10_POWER_MULTIPLIER = ATM10_POWER_PRODUCTION_MULTIPLIER * ATM10_REACTOR_POWER_MULTIPLIER

# This defaults to 1.0 because it reproduces the measured ATM10 9x9x9
# checkerboard result (~530 kFE/t, ~0.87 mB/t). It is exposed as a CLI option
# in case your installed ER2 build behaves differently.
DEFAULT_VARIANT_EFFICIENCY = 1.0

# ---------------------------------------------------------------------------
# Unobtainium moderator values used by ATM10 / ER2 registrations
# ---------------------------------------------------------------------------

UNOBTAINIUM_ABSORPTION = 0.972
UNOBTAINIUM_HEAT_EFFICIENCY = 0.91
UNOBTAINIUM_MODERATION = 3.074
UNOBTAINIUM_CONDUCTIVITY = 5.0

CASING_CONDUCTIVITY = 0.6

ALL_MASK = (1 << CELL_COUNT) - 1


@dataclass
class Result:
    score: float
    efficiency_fe_per_mb: float
    power_fe_t: float
    fuel_mb_t: float
    fuel_heat_c: float
    reactor_heat_c: float
    fertility: float
    rod_columns: int
    rod_blocks: int
    mask: int
    iterations: int = 0

    def serializable(self):
        d = asdict(self)
        d["layout"] = render_layout(self.mask)
        return d


def bit_index(x: int, z: int) -> int:
    return z * W + x


def has_rod(mask: int, x: int, z: int) -> bool:
    return bool(mask & (1 << bit_index(x, z)))


def positions(mask: int) -> list[tuple[int, int]]:
    out = []
    for z in range(D):
        for x in range(W):
            if has_rod(mask, x, z):
                out.append((x, z))
    return out


def rod_count(mask: int) -> int:
    return mask.bit_count()


def render_layout(mask: int) -> str:
    rows = []
    for z in range(D):
        rows.append(" ".join("R" if has_rod(mask, x, z) else "U" for x in range(W)))
    return "\n".join(rows)


def checkerboard_mask() -> int:
    m = 0
    for z in range(D):
        for x in range(W):
            if (x + z) % 2 == 0:
                m |= 1 << bit_index(x, z)
    return m


def spaced_3x3_mask() -> int:
    m = 0
    for z in (0, 3, 6):
        for x in (0, 3, 6):
            m |= 1 << bit_index(x, z)
    return m


def spaced_4x4_mask() -> int:
    m = 0
    for z in (0, 2, 4, 6):
        for x in (0, 2, 4, 6):
            m |= 1 << bit_index(x, z)
    return m


def cross_mask() -> int:
    m = 0
    for z in range(D):
        for x in range(W):
            if x == 3 or z == 3:
                m |= 1 << bit_index(x, z)
    return m


def random_mask(rng: random.Random, min_rods: int, max_rods: int) -> int:
    # Bias toward the middle/lower end where fuel-efficient layouts generally live.
    a = rng.randint(min_rods, max_rods)
    b = rng.randint(min_rods, max_rods)
    n = min(a, b)
    chosen = rng.sample(range(CELL_COUNT), n)
    m = 0
    for i in chosen:
        m |= 1 << i
    return m


def mutate(mask: int, rng: random.Random, min_rods: int, max_rods: int) -> int:
    n = rod_count(mask)
    mode = rng.random()

    if mode < 0.45 and min_rods <= n <= max_rods:
        # Move one rod: preserves rod count while changing geometry.
        on = [i for i in range(CELL_COUNT) if mask & (1 << i)]
        off = [i for i in range(CELL_COUNT) if not mask & (1 << i)]
        if on and off:
            mask ^= 1 << rng.choice(on)
            mask ^= 1 << rng.choice(off)
            return mask

    # Flip 1-3 cells.
    flips = 1 if mode < 0.75 else (2 if mode < 0.93 else 3)
    for _ in range(flips):
        i = rng.randrange(CELL_COUNT)
        new_mask = mask ^ (1 << i)
        nr = rod_count(new_mask)
        if min_rods <= nr <= max_rods:
            mask = new_mask

    return mask


def heat_transfer_coefficient(mask: int) -> float:
    """
    Mirrors ReactorFuelRodEntity.getHeatTransferRate():
      - adjacent fuel rod: no transfer
      - Unobtainium neighbor: conductivity 5.0
      - reactor casing boundary: conductivity 0.6
    Then multiplied by the seven identical vertical layers.
    """
    total_per_layer = 0.0

    for z in range(D):
        for x in range(W):
            if not has_rod(mask, x, z):
                continue

            for dx, dz in DIRECTIONS:
                nx, nz = x + dx, z + dz

                if nx < 0 or nx >= W or nz < 0 or nz >= D:
                    total_per_layer += CASING_CONDUCTIVITY
                elif has_rod(mask, nx, nz):
                    # Fuel-to-fuel face does not conduct into the reactor environment.
                    pass
                else:
                    total_per_layer += UNOBTAINIUM_CONDUCTIVITY

    return total_per_layer * H


def source_paths(mask: int, rods: list[tuple[int, int]]):
    """
    Precompute up-to-4-block horizontal radiation paths for each source column.
    Entries are True for a fuel rod and False for Unobtainium.
    A path stops at the reactor casing.
    """
    all_paths = []

    for sx, sz in rods:
        source = []
        for dx, dz in DIRECTIONS:
            path = []
            x, z = sx, sz

            for _ in range(4):
                x += dx
                z += dz

                if x < 0 or x >= W or z < 0 or z >= D:
                    break

                path.append(has_rod(mask, x, z))

            source.append(tuple(path))
        all_paths.append(tuple(source))

    return tuple(all_paths)


def simulate(
    mask: int,
    *,
    insertion: float,
    fill: float,
    variant_efficiency: float,
    max_ticks: int,
    min_ticks: int,
    sample_ticks: int,
) -> Result | None:
    """
    Simulate the current ER2 1.21 reactor loop:
      1. irradiate from exactly one fuel-rod block
      2. decay fertility
      3. transfer fuel heat -> reactor
      4. transfer reactor heat -> passive coolant / generate FE
      5. passive heat loss

    The 7 vertical blocks in a rod column have identical horizontal geometry,
    so cycling columns is equivalent for steady-state behavior while avoiding
    redundant vertical-coordinate bookkeeping.
    """
    ncols = rod_count(mask)
    if ncols <= 0:
        return None

    rods = positions(mask)
    paths = source_paths(mask, rods)

    num_fuel_rod_blocks = ncols * H
    full_fuel_amount = num_fuel_rod_blocks * FUEL_CAPACITY_PER_ROD_BLOCK
    base_fuel_amount = full_fuel_amount * fill

    if base_fuel_amount <= 0:
        return None

    htc = heat_transfer_coefficient(mask)

    control_rod_modifier = 1.0 - insertion
    raw_rad_intensity = base_fuel_amount * FISSION_EVENTS_PER_FUEL_UNIT * control_rod_modifier

    scaled_rad_intensity = raw_rad_intensity ** FUEL_REACTIVITY
    scaled_rad_intensity = (
        (scaled_rad_intensity / ncols) ** FUEL_REACTIVITY
    ) * ncols
    scaled_rad_intensity *= control_rod_modifier

    fuel_moderation = (
        FUEL_MODERATION_FACTOR
        + FUEL_MODERATION_FACTOR * insertion
        + insertion
    )

    fuel_heat = 0.0
    reactor_heat = 0.0
    fertility = 1.0

    # Running sample window.
    power_window = [0.0] * sample_ticks
    fuel_window = [0.0] * sample_ticks
    widx = 0
    wcount = 0

    # Convergence snapshots.
    last_snapshot = None
    stable_checks = 0

    for tick in range(max_ticks):
        src_i = tick % ncols

        # ---- ER2 radiate() ----
        radiation_penalty_base = math.exp(-15.0 * math.exp(-0.0025 * fuel_heat))
        rad_hardness = 0.2 + 0.8 * radiation_penalty_base

        fertility_modifier = 1.0 if fertility <= 1.0 else math.log10(fertility) + 1.0
        fuel_usage = (
            FUEL_UNITS_PER_FISSION_EVENT
            * raw_rad_intensity
            / fertility_modifier
            * ATM10_FUEL_USAGE_MULTIPLIER
        )

        effective_rad_intensity = scaled_rad_intensity * (
            1.0 + (-0.95 * math.exp(-10.0 * math.exp(-0.0012 * fuel_heat)))
        )

        fuel_energy_absorption = ENERGY_PER_RADIATION_UNIT * effective_rad_intensity
        environment_energy_absorption = 0.0
        fuel_absorbed_radiation = 0.0

        per_direction_intensity = effective_rad_intensity * 0.25

        for path in paths[src_i]:
            intensity = per_direction_intensity
            hardness = rad_hardness

            for is_fuel in path:
                if intensity <= 0.0001:
                    break

                if is_fuel:
                    # ReactorFuelRodEntity.moderateRadiation()
                    fuel_heat_response = 1.0 - (
                        0.95 * math.exp(-10.0 * math.exp(-0.0022 * fuel_heat))
                    )

                    base_absorption = fuel_heat_response * (
                        1.0 - hardness / FUEL_HARDNESS_DIVISOR
                    )
                    scaled_absorption = min(
                        1.0,
                        base_absorption * FUEL_ABSORPTION_COEFF,
                    )

                    control_rod_bonus = (
                        (1.0 - scaled_absorption) * insertion * 0.5
                    )
                    control_rod_penalty = (
                        scaled_absorption * insertion * 0.5
                    )

                    radiation_absorbed = (
                        scaled_absorption + control_rod_bonus
                    ) * intensity
                    fertility_absorbed = (
                        scaled_absorption - control_rod_penalty
                    ) * intensity

                    intensity = max(0.0, intensity - radiation_absorbed)
                    hardness /= fuel_moderation

                    fuel_energy_absorption += (
                        radiation_absorbed * ENERGY_PER_RADIATION_UNIT
                    )
                    fuel_absorbed_radiation += fertility_absorbed

                else:
                    # MultiblockReactor.applyModerator()
                    radiation_absorbed = (
                        intensity
                        * UNOBTAINIUM_ABSORPTION
                        * (1.0 - hardness)
                    )

                    intensity = max(0.0, intensity - radiation_absorbed)
                    hardness /= UNOBTAINIUM_MODERATION

                    environment_energy_absorption += (
                        UNOBTAINIUM_HEAT_EFFICIENCY
                        * radiation_absorbed
                        * ENERGY_PER_RADIATION_UNIT
                    )

        fertility += fuel_absorbed_radiation

        # Irradiation heat is distributed over the full reactor fuel/environment volume.
        fuel_heat += (
            fuel_energy_absorption
            / (num_fuel_rod_blocks * ENERGY_PER_C_PER_UNIT_VOLUME)
        )
        reactor_heat += (
            environment_energy_absorption
            / (REACTOR_VOLUME * ENERGY_PER_C_PER_UNIT_VOLUME)
        )

        # ---- ER2 performRadiationDecay() ----
        fertility = max(
            0.0,
            fertility - max(0.1, fertility / 20.0),
        )

        # ---- transferHeatBetweenFuelAndReactor() ----
        temperature_diff = fuel_heat - reactor_heat
        if temperature_diff > 0.01:
            energy_transferred = temperature_diff * htc

            fuel_energy = (
                fuel_heat
                * num_fuel_rod_blocks
                * ENERGY_PER_C_PER_UNIT_VOLUME
                - energy_transferred
            )
            reactor_energy = (
                reactor_heat
                * REACTOR_VOLUME
                * ENERGY_PER_C_PER_UNIT_VOLUME
                + energy_transferred
            )

            fuel_heat = fuel_energy / (
                num_fuel_rod_blocks * ENERGY_PER_C_PER_UNIT_VOLUME
            )
            reactor_heat = reactor_energy / (
                REACTOR_VOLUME * ENERGY_PER_C_PER_UNIT_VOLUME
            )

        # ---- transferHeatBetweenReactorAndCoolant(), passive mode ----
        temperature_diff = reactor_heat - AMBIENT_TEMP
        power = 0.0

        if temperature_diff > 0.01:
            energy_transferred = (
                temperature_diff * REACTOR_TO_COOLANT_COEFF
            )

            reactor_energy = (
                reactor_heat
                * REACTOR_VOLUME
                * ENERGY_PER_C_PER_UNIT_VOLUME
            )

            energy_transferred *= PASSIVE_COOLING_TRANSFER_EFFICIENCY

            power = (
                energy_transferred
                * PASSIVE_COOLING_POWER_EFFICIENCY
                * ATM10_POWER_MULTIPLIER
                * variant_efficiency
            )

            reactor_energy -= energy_transferred
            reactor_heat = reactor_energy / (
                REACTOR_VOLUME * ENERGY_PER_C_PER_UNIT_VOLUME
            )

        # ---- performPassiveHeatLoss() ----
        temperature_diff = reactor_heat - AMBIENT_TEMP

        if temperature_diff > 0.000001:
            energy_lost = max(
                1.0,
                temperature_diff * REACTOR_HEAT_LOSS_COEFF,
            )

            reactor_energy = max(
                0.0,
                reactor_heat
                * REACTOR_VOLUME
                * ENERGY_PER_C_PER_UNIT_VOLUME
                - energy_lost,
            )

            reactor_heat = reactor_energy / (
                REACTOR_VOLUME * ENERGY_PER_C_PER_UNIT_VOLUME
            )

        fuel_heat = max(0.0, fuel_heat)
        reactor_heat = max(0.0, reactor_heat)

        # Record last N ticks.
        power_window[widx] = power
        fuel_window[widx] = fuel_usage
        widx = (widx + 1) % sample_ticks
        wcount = min(sample_ticks, wcount + 1)

        # Convergence check every 250 ticks after minimum warmup.
        if tick + 1 >= min_ticks and (tick + 1) % 250 == 0:
            snapshot = (fuel_heat, reactor_heat, fertility)

            if last_snapshot is not None:
                # Tight enough to rank layouts, while still fast enough to brute-force.
                delta = max(
                    abs(snapshot[0] - last_snapshot[0]) / max(1.0, abs(snapshot[0])),
                    abs(snapshot[1] - last_snapshot[1]) / max(1.0, abs(snapshot[1])),
                    abs(snapshot[2] - last_snapshot[2]) / max(1.0, abs(snapshot[2])),
                )

                if delta < 2e-5:
                    stable_checks += 1
                else:
                    stable_checks = 0

                if stable_checks >= 2 and wcount >= sample_ticks:
                    break

            last_snapshot = snapshot

    if wcount == 0:
        return None

    avg_power = sum(power_window[:wcount]) / wcount
    avg_fuel = sum(fuel_window[:wcount]) / wcount

    if avg_fuel <= 0.0:
        return None

    efficiency = avg_power / avg_fuel

    return Result(
        score=efficiency,
        efficiency_fe_per_mb=efficiency,
        power_fe_t=avg_power,
        fuel_mb_t=avg_fuel,
        fuel_heat_c=fuel_heat,
        reactor_heat_c=reactor_heat,
        fertility=fertility,
        rod_columns=ncols,
        rod_blocks=ncols * H,
        mask=mask,
    )


def score_result(result: Result | None, min_power: float) -> float:
    if result is None:
        return float("-inf")
    if result.power_fe_t < min_power:
        return float("-inf")
    return result.efficiency_fe_per_mb


def worker(args_tuple):
    (
        worker_id,
        seconds,
        min_rods,
        max_rods,
        insertion,
        fill,
        variant_efficiency,
        min_power,
        search_max_ticks,
        search_min_ticks,
        sample_ticks,
        seed,
    ) = args_tuple

    rng = random.Random(seed + worker_id * 1_000_003)
    deadline = time.perf_counter() + seconds

    seed_masks = [
        checkerboard_mask(),
        spaced_3x3_mask(),
        spaced_4x4_mask(),
        cross_mask(),
    ]

    # Filter seed layouts to allowed rod-count range.
    seed_masks = [
        m for m in seed_masks
        if min_rods <= rod_count(m) <= max_rods
    ]

    local_best = None
    local_best_score = float("-inf")
    iterations = 0

    # Evaluate deterministic seeds first.
    for mask in seed_masks:
        result = simulate(
            mask,
            insertion=insertion,
            fill=fill,
            variant_efficiency=variant_efficiency,
            max_ticks=search_max_ticks,
            min_ticks=search_min_ticks,
            sample_ticks=sample_ticks,
        )
        iterations += 1
        s = score_result(result, min_power)
        if s > local_best_score:
            local_best = result
            local_best_score = s

    # If every seed misses a power target, still give mutation search a starting point.
    parent_mask = (
        local_best.mask
        if local_best is not None
        else random_mask(rng, min_rods, max_rods)
    )

    while time.perf_counter() < deadline:
        r = rng.random()

        if r < 0.72:
            candidate = mutate(parent_mask, rng, min_rods, max_rods)
        elif r < 0.90 and local_best is not None:
            # More aggressive mutation around the current champion.
            candidate = local_best.mask
            for _ in range(rng.randint(2, 5)):
                candidate = mutate(candidate, rng, min_rods, max_rods)
        else:
            candidate = random_mask(rng, min_rods, max_rods)

        if candidate == 0:
            continue

        result = simulate(
            candidate,
            insertion=insertion,
            fill=fill,
            variant_efficiency=variant_efficiency,
            max_ticks=search_max_ticks,
            min_ticks=search_min_ticks,
            sample_ticks=sample_ticks,
        )
        iterations += 1

        s = score_result(result, min_power)

        if s > local_best_score:
            local_best = result
            local_best_score = s
            parent_mask = candidate
        else:
            # Occasionally accept a non-best candidate as the next parent so the
            # walk can escape local maxima without corrupting the saved champion.
            if result is not None and rng.random() < 0.025:
                parent_mask = candidate

    if local_best is None:
        # Return a diagnostic result rather than crashing the parent.
        fallback = simulate(
            checkerboard_mask(),
            insertion=insertion,
            fill=fill,
            variant_efficiency=variant_efficiency,
            max_ticks=search_max_ticks,
            min_ticks=search_min_ticks,
            sample_ticks=sample_ticks,
        )
        local_best = fallback

    local_best.iterations = iterations
    return local_best


def print_result(title: str, result: Result):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)
    print(render_layout(result.mask))
    print()
    print(f"Rod columns : {result.rod_columns}")
    print(f"Fuel blocks : {result.rod_blocks}")
    print(f"Power       : {result.power_fe_t:,.2f} FE/t")
    print(f"Fuel usage  : {result.fuel_mb_t:,.6f} mB/t")
    print(f"Efficiency  : {result.efficiency_fe_per_mb:,.2f} FE/mB")
    print(f"Fuel heat   : {result.fuel_heat_c:,.2f} C")
    print(f"Reactor heat: {result.reactor_heat_c:,.2f} C")
    print(f"Fertility   : {result.fertility:,.2f}")
    print(f"Mask        : 0x{result.mask:013x}")


def save_result(path_prefix: str, result: Result, settings: dict):
    json_path = Path(f"{path_prefix}.json")
    txt_path = Path(f"{path_prefix}.txt")

    payload = {
        "settings": settings,
        "result": result.serializable(),
    }

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    txt_path.write_text(
        render_layout(result.mask)
        + "\n\n"
        + f"Rod columns: {result.rod_columns}\n"
        + f"Fuel blocks: {result.rod_blocks}\n"
        + f"Power: {result.power_fe_t:.6f} FE/t\n"
        + f"Fuel usage: {result.fuel_mb_t:.9f} mB/t\n"
        + f"Efficiency: {result.efficiency_fe_per_mb:.6f} FE/mB\n"
        + f"Fuel heat: {result.fuel_heat_c:.6f} C\n"
        + f"Reactor heat: {result.reactor_heat_c:.6f} C\n"
        + f"Fertility: {result.fertility:.6f}\n",
        encoding="utf-8",
    )

    return json_path, txt_path


def parse_args():
    cpu = os.cpu_count() or 1

    p = argparse.ArgumentParser(
        description="Brute-force / evolutionary ATM10 Extreme Reactors 2 9x9x9 layouts."
    )

    p.add_argument("--threads", type=int, default=cpu,
                   help=f"CPU workers/processes to use (default: {cpu})")
    p.add_argument("--seconds", type=float, default=60.0,
                   help="Search time per worker in seconds (default: 60)")
    p.add_argument("--min-rods", type=int, default=1,
                   help="Minimum fuel-rod columns (default: 1)")
    p.add_argument("--max-rods", type=int, default=35,
                   help="Maximum fuel-rod columns (default: 35)")
    p.add_argument("--min-power", type=float, default=0.0,
                   help="Reject layouts below this FE/t; e.g. 500000")
    p.add_argument("--insertion", type=float, default=0.0,
                   help="Control-rod insertion percent, 0..100 (default: 0)")
    p.add_argument("--fill", type=float, default=1.0,
                   help="Fuel fill fraction, 0..1 (default: 1.0)")
    p.add_argument("--variant-efficiency", type=float, default=DEFAULT_VARIANT_EFFICIENCY,
                   help="Final generation efficiency multiplier (default: 1.0, ATM10-calibrated)")
    p.add_argument("--search-max-ticks", type=int, default=4500,
                   help="Maximum sim ticks per candidate during search (default: 4500)")
    p.add_argument("--search-min-ticks", type=int, default=1500,
                   help="Minimum sim ticks per candidate before convergence checks (default: 1500)")
    p.add_argument("--final-ticks", type=int, default=20000,
                   help="High-precision reevaluation ticks for winner (default: 20000)")
    p.add_argument("--sample-ticks", type=int, default=500,
                   help="Averaging window in ticks (default: 500)")
    p.add_argument("--seed", type=int, default=1337,
                   help="Base RNG seed (default: 1337)")
    p.add_argument("--output", default="best_reactor",
                   help="Output prefix for .json and .txt files (default: best_reactor)")
    p.add_argument("--calibrate", action="store_true",
                   help="Only simulate the 25-column checkerboard sanity check and exit")

    return p.parse_args()


def main():
    args = parse_args()

    if args.threads < 1:
        raise SystemExit("--threads must be >= 1")
    if args.seconds <= 0:
        raise SystemExit("--seconds must be > 0")
    if not (1 <= args.min_rods <= CELL_COUNT):
        raise SystemExit("--min-rods must be between 1 and 49")
    if not (1 <= args.max_rods <= CELL_COUNT):
        raise SystemExit("--max-rods must be between 1 and 49")
    if args.min_rods > args.max_rods:
        raise SystemExit("--min-rods cannot exceed --max-rods")
    if not (0.0 <= args.insertion <= 100.0):
        raise SystemExit("--insertion must be between 0 and 100")
    if not (0.0 < args.fill <= 1.0):
        raise SystemExit("--fill must be > 0 and <= 1")

    insertion = args.insertion / 100.0

    print("ATM10 Extreme Reactors 2 optimizer")
    print("----------------------------------")
    print(f"Workers       : {args.threads}")
    print(f"Runtime       : {args.seconds:.1f} s")
    print(f"Rod range     : {args.min_rods}..{args.max_rods} columns")
    print(f"Min power     : {args.min_power:,.0f} FE/t")
    print(f"Rod insertion : {args.insertion:.1f}%")
    print(f"Fuel fill     : {args.fill * 100:.1f}%")
    print(f"Variant eff.  : {args.variant_efficiency:.4f}")
    print("Moderator     : Unobtainium only")
    print()

    # Calibration case from the conversation: 25-column 7x7 checkerboard.
    checker = simulate(
        checkerboard_mask(),
        insertion=insertion,
        fill=args.fill,
        variant_efficiency=args.variant_efficiency,
        max_ticks=max(args.final_ticks, 12000),
        min_ticks=6000,
        sample_ticks=max(args.sample_ticks, 1000),
    )
    print_result("CHECKERBOARD SANITY CHECK", checker)

    if args.calibrate:
        return

    worker_args = [
        (
            i,
            args.seconds,
            args.min_rods,
            args.max_rods,
            insertion,
            args.fill,
            args.variant_efficiency,
            args.min_power,
            args.search_max_ticks,
            args.search_min_ticks,
            args.sample_ticks,
            args.seed,
        )
        for i in range(args.threads)
    ]

    print()
    print("Searching...")

    # spawn works reliably on Windows, Linux, and macOS.
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.threads) as pool:
        worker_results = pool.map(worker, worker_args)

    total_iterations = sum(r.iterations for r in worker_results if r is not None)
    winner = max(
        (r for r in worker_results if r is not None),
        key=lambda r: score_result(r, args.min_power),
    )

    # Re-evaluate the winning geometry at much higher precision.
    final = simulate(
        winner.mask,
        insertion=insertion,
        fill=args.fill,
        variant_efficiency=args.variant_efficiency,
        max_ticks=args.final_ticks,
        min_ticks=min(8000, max(2000, args.final_ticks // 3)),
        sample_ticks=max(args.sample_ticks, 1000),
    )
    final.iterations = total_iterations

    print_result("BEST LAYOUT FOUND", final)
    print(f"Candidates tested across workers: {total_iterations:,}")

    settings = {
        "threads": args.threads,
        "seconds": args.seconds,
        "min_rods": args.min_rods,
        "max_rods": args.max_rods,
        "min_power": args.min_power,
        "insertion_percent": args.insertion,
        "fuel_fill": args.fill,
        "variant_efficiency": args.variant_efficiency,
        "search_max_ticks": args.search_max_ticks,
        "search_min_ticks": args.search_min_ticks,
        "final_ticks": args.final_ticks,
        "sample_ticks": args.sample_ticks,
        "seed": args.seed,
        "atm10_fuel_multiplier": ATM10_FUEL_USAGE_MULTIPLIER,
        "atm10_power_multiplier": ATM10_POWER_MULTIPLIER,
        "unobtainium": {
            "absorption": UNOBTAINIUM_ABSORPTION,
            "heat_efficiency": UNOBTAINIUM_HEAT_EFFICIENCY,
            "moderation": UNOBTAINIUM_MODERATION,
            "conductivity": UNOBTAINIUM_CONDUCTIVITY,
        },
        "search_objective": "maximize FE/mB subject to min_power",
    }

    json_path, txt_path = save_result(args.output, final, settings)

    print()
    print(f"Saved JSON : {json_path.resolve()}")
    print(f"Saved layout: {txt_path.resolve()}")


if __name__ == "__main__":
    mp.freeze_support()
    main()

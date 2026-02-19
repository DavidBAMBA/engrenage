#!/usr/bin/env python3
"""
Benchmark: Numba vs JAX — BH and Oscillaton Evolution

Compares wall-clock time for BH and Oscillaton (BSSN + scalar field) evolution
at different grid resolutions using both Numba and JAX backends.
Both use manual RK4 with fixed step (dt = 0.4 * min_dr).

Usage:
    python examples/benchmark_bh_oscillaton.py

Environment variables:
    T_FINAL:       Evolution time (default: 2.0)
    RESOLUTIONS:   Comma-separated grid sizes (default: "300,1000,4000,10000")
"""

import os
import sys
import time
import contextlib
import numpy as np

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# =============================================================================
# Configuration — edit these or set via environment variables
# =============================================================================
T_FINAL = float(os.environ.get("T_FINAL", 2.0))
RESOLUTIONS = [int(x) for x in
               os.environ.get("RESOLUTIONS", "300,1000,4000,10000").split(",")]

SCALAR_MU = 1.0
ETA = 1.0
SIGMA_BASE = 1.0
N_BENCH_STEPS = 10    # steps to average for per-step timing

BH_R_MAX = 96.0       # BH domain size (fixed)
OSC_R_MAX_BASE = 64.0  # Oscillaton base domain
OSC_DR_MIN = 0.012     # keep dr > 0.01 (CSV data spacing) for oscillaton IC

# =============================================================================
# Imports
# =============================================================================
from source.core.grid import Grid
from source.core.spacing import LinearSpacing, NUM_GHOSTS
from source.core.statevector import StateVector
from source.matter.scalarmatter import ScalarMatter
from source.backgrounds.sphericalbackground import FlatSphericalBackground
from source.bssn.bssnstatevariables import (
    NUM_BSSN_VARS, BSSN_PARITY, BSSN_ASYMP_POWER, BSSN_ASYMP_OFFSET,
)
from source.initialdata.bhinitialconditions import get_initial_state as get_bh_ic
from source.core.rhsevolution import get_rhs


# =============================================================================
# Helpers
# =============================================================================
@contextlib.contextmanager
def pushd(path):
    """Temporarily change working directory."""
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


class _DummyBar:
    def update(self, n):
        pass


def setup_bh(N):
    """Create BH problem at resolution N."""
    matter = ScalarMatter(SCALAR_MU)
    sv = StateVector(matter)
    spacing = LinearSpacing(N, BH_R_MAX)
    grid = Grid(spacing, sv)
    bg = FlatSphericalBackground(grid.r)
    state = get_bh_ic(grid, bg)
    return grid, bg, matter, sv, state


def setup_oscillaton(N):
    """Create Oscillaton problem at resolution N.
    Adjusts r_max for high N to keep dr > 0.01 (CSV data constraint)."""
    r_max = max(OSC_R_MAX_BASE, OSC_DR_MIN * N)
    matter = ScalarMatter(SCALAR_MU)
    sv = StateVector(matter)
    spacing = LinearSpacing(N, r_max)
    grid = Grid(spacing, sv)
    bg = FlatSphericalBackground(grid.r)
    # Oscillaton IC loads CSV with relative paths — needs CWD = examples/
    from source.initialdata.oscillatoninitialconditions import (
        get_initial_state as get_osc_ic,
    )
    with pushd(script_dir):
        state = get_osc_ic(grid, bg)
    return grid, bg, matter, sv, state, r_max


# =============================================================================
# Numba benchmark
# =============================================================================
def bench_numba(label, grid, bg, matter, state_flat):
    """Benchmark Numba RK4 for a given scenario. Returns timing dict."""
    N = grid.N
    dt = 0.4 * float(grid.min_dr)
    n_steps = int(np.ceil(T_FINAL / dt))
    dummy = _DummyBar()

    def rhs(s):
        return get_rhs(0.0, s, grid, bg, matter, dummy, [0.0, 1e10])

    def rk4(s, h):
        k1 = rhs(s)
        k2 = rhs(s + 0.5 * h * k1)
        k3 = rhs(s + 0.5 * h * k2)
        k4 = rhs(s + h * k3)
        return s + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # Warmup (triggers Numba JIT on first call)
    print(f"  [{label}/Numba] N={N}: warmup...", end="", flush=True)
    t0 = time.time()
    _ = rhs(state_flat)
    t_warm = time.time() - t0
    print(f" {t_warm:.1f}s")

    # Per-step benchmark
    state = state_flat.copy()
    t0 = time.time()
    for _ in range(N_BENCH_STEPS):
        state = rk4(state, dt)
    t_per_step = (time.time() - t0) / N_BENCH_STEPS

    # Full evolution
    est = t_per_step * n_steps
    print(f"  [{label}/Numba] N={N}: evolving T={T_FINAL} "
          f"({n_steps} steps, est ~{est:.0f}s)...")
    state = state_flat.copy()
    t_cur = 0.0
    t0 = time.time()
    for step in range(1, n_steps + 1):
        h = min(dt, T_FINAL - t_cur)
        if h <= 0:
            break
        state = rk4(state, h)
        t_cur += h
        if step % max(1, n_steps // 5) == 0:
            el = time.time() - t0
            print(f"    {step / n_steps * 100:.0f}%  t={t_cur:.2f}/{T_FINAL}  "
                  f"elapsed={el:.1f}s")
    t_total = time.time() - t0

    print(f"  [{label}/Numba] N={N}: done -- "
          f"{t_per_step * 1000:.1f} ms/step, total={t_total:.1f}s")
    return dict(N=N, backend="Numba", scenario=label,
                dt=dt, n_steps=n_steps, warmup_s=t_warm,
                per_step_ms=t_per_step * 1000, total_s=t_total)


# =============================================================================
# JAX benchmark
# =============================================================================
def bench_jax(label, grid, bg, matter, sv, state_flat):
    """Benchmark JAX RK4 for a given scenario. Returns timing dict."""
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from source.bssn.jax.bssngeometry import (
        build_bssn_background, build_derivative_stencils,
    )
    from source.core.rhsevolution_jax import get_rhs_bssn_scalar_jax
    from source.bssn.jax.boundaries_jax import fill_bssn_boundaries_jax

    N = grid.N
    num_vars = grid.NUM_VARS
    dt_val = 0.4 * float(grid.min_dr)
    n_steps = int(np.ceil(T_FINAL / dt_val))

    # Transfer to JAX
    state_2d = jnp.array(state_flat.reshape(num_vars, -1))
    bssn_bg = build_bssn_background(grid, bg)
    deriv_stencils = build_derivative_stencils(grid)
    dr = jnp.array(grid.dr)

    parity_j = jnp.array(np.concatenate([
        np.array(BSSN_PARITY, dtype=np.float64),
        np.ones(num_vars - NUM_BSSN_VARS)]))
    asymp_pow_j = jnp.array(np.concatenate([
        np.array(BSSN_ASYMP_POWER, dtype=np.float64),
        np.zeros(num_vars - NUM_BSSN_VARS)]))
    asymp_off_j = jnp.array(np.concatenate([
        np.array(BSSN_ASYMP_OFFSET, dtype=np.float64),
        np.zeros(num_vars - NUM_BSSN_VARS)]))

    @jax.jit
    def apply_bcs(s):
        return fill_bssn_boundaries_jax(
            s, bssn_bg.r, NUM_GHOSTS,
            parity_j, asymp_pow_j, asymp_off_j)

    @jax.jit
    def rhs_fn(s):
        return get_rhs_bssn_scalar_jax(
            s, bssn_bg, deriv_stencils, dr,
            NUM_GHOSTS, num_vars, SIGMA_BASE, SCALAR_MU, ETA,
            outer_bc_type="asymptotic", fix_shift=False)

    @jax.jit
    def rk4_step(s, dt):
        k1 = rhs_fn(s)
        k2 = rhs_fn(s + 0.5 * dt * k1)
        k3 = rhs_fn(s + 0.5 * dt * k2)
        k4 = rhs_fn(s + dt * k3)
        return apply_bcs(s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4))

    dt_jax = jnp.float64(dt_val)

    # JIT compilation
    print(f"  [{label}/JAX]   N={N}: JIT compiling...", end="", flush=True)
    t0 = time.time()
    _ = rk4_step(state_2d, dt_jax).block_until_ready()
    t_jit = time.time() - t0
    print(f" {t_jit:.1f}s")

    # Per-step benchmark
    state = state_2d
    t0 = time.time()
    for _ in range(N_BENCH_STEPS):
        state = rk4_step(state, dt_jax)
    state.block_until_ready()
    t_per_step = (time.time() - t0) / N_BENCH_STEPS

    # Full evolution
    print(f"  [{label}/JAX]   N={N}: evolving T={T_FINAL} "
          f"({n_steps} steps)...")
    state = state_2d
    t_cur = 0.0
    t0 = time.time()
    for step in range(1, n_steps + 1):
        h = min(dt_val, T_FINAL - t_cur)
        if h <= 0:
            break
        state = rk4_step(state, jnp.float64(h))
        t_cur += h
        if step % max(1, n_steps // 5) == 0:
            state.block_until_ready()
            el = time.time() - t0
            print(f"    {step / n_steps * 100:.0f}%  t={t_cur:.2f}/{T_FINAL}  "
                  f"elapsed={el:.1f}s")
    state.block_until_ready()
    t_total = time.time() - t0

    print(f"  [{label}/JAX]   N={N}: done -- "
          f"{t_per_step * 1000:.1f} ms/step, total={t_total:.1f}s "
          f"(JIT={t_jit:.1f}s)")
    return dict(N=N, backend="JAX", scenario=label,
                dt=dt_val, n_steps=n_steps, jit_s=t_jit,
                per_step_ms=t_per_step * 1000, total_s=t_total)


# =============================================================================
# Plotting
# =============================================================================
def plot_results(bh_numba, bh_jax, osc_numba, osc_jax, path):
    """Generate 2x2 benchmark comparison plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Ns = [r["N"] for r in bh_numba]
    bn = [r["per_step_ms"] for r in bh_numba]
    bj = [r["per_step_ms"] for r in bh_jax]
    on = [r["per_step_ms"] for r in osc_numba]
    oj = [r["per_step_ms"] for r in osc_jax]

    bh_spdup = [n / j for n, j in zip(bn, bj)]
    osc_spdup = [n / j for n, j in zip(on, oj)]

    bt_n = [r["total_s"] for r in bh_numba]
    bt_j = [r["total_s"] for r in bh_jax]
    ot_n = [r["total_s"] for r in osc_numba]
    ot_j = [r["total_s"] for r in osc_jax]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Benchmark: Numba vs JAX -- BH & Oscillaton (T={T_FINAL})",
                 fontsize=14, fontweight="bold")

    # -- (0,0) Per-step time --
    ax = axes[0, 0]
    ax.plot(Ns, bn, "o-",  color="tab:blue",   lw=2, ms=7, label="BH Numba")
    ax.plot(Ns, bj, "s-",  color="tab:orange",  lw=2, ms=7, label="BH JAX")
    ax.plot(Ns, on, "o--", color="tab:blue",   lw=1.5, ms=5, alpha=0.6,
            label="Osc Numba")
    ax.plot(Ns, oj, "s--", color="tab:orange",  lw=1.5, ms=5, alpha=0.6,
            label="Osc JAX")
    ax.set_xlabel("Grid points N")
    ax.set_ylabel("ms / RK4 step")
    ax.set_title("Per-step time")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    # -- (0,1) Total evolution time --
    ax = axes[0, 1]
    ax.plot(Ns, bt_n, "o-",  color="tab:blue",   lw=2, ms=7, label="BH Numba")
    ax.plot(Ns, bt_j, "s-",  color="tab:orange",  lw=2, ms=7, label="BH JAX")
    ax.plot(Ns, ot_n, "o--", color="tab:blue",   lw=1.5, ms=5, alpha=0.6,
            label="Osc Numba")
    ax.plot(Ns, ot_j, "s--", color="tab:orange",  lw=1.5, ms=5, alpha=0.6,
            label="Osc JAX")
    ax.set_xlabel("Grid points N")
    ax.set_ylabel("Total time (s)")
    ax.set_title(f"Total evolution (T={T_FINAL})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    # -- (1,0) Speedup --
    ax = axes[1, 0]
    ax.plot(Ns, bh_spdup, "D-", color="tab:green",  lw=2, ms=8,
            label="BH speedup")
    ax.plot(Ns, osc_spdup, "^-", color="tab:purple", lw=2, ms=8,
            label="Oscillaton speedup")
    ax.axhline(1, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Grid points N")
    ax.set_ylabel("Speedup (Numba / JAX)")
    ax.set_title("Per-step speedup")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    for i, n in enumerate(Ns):
        ax.annotate(f"{bh_spdup[i]:.1f}x", (n, bh_spdup[i]),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9, color="tab:green")
        ax.annotate(f"{osc_spdup[i]:.1f}x", (n, osc_spdup[i]),
                    textcoords="offset points", xytext=(0, -15),
                    ha="center", fontsize=9, color="tab:purple")

    # -- (1,1) Summary table --
    ax = axes[1, 1]
    ax.axis("off")
    rows = []
    for i in range(len(Ns)):
        rows.append([
            f"{Ns[i]}",
            f"{bn[i]:.1f}", f"{bj[i]:.1f}", f"{bh_spdup[i]:.1f}x",
            f"{on[i]:.1f}", f"{oj[i]:.1f}", f"{osc_spdup[i]:.1f}x",
        ])
    col_labels = ["N",
                  "BH\nNumba", "BH\nJAX", "BH\nSpeedup",
                  "Osc\nNumba", "Osc\nJAX", "Osc\nSpeedup"]
    tbl = ax.table(cellText=rows, colLabels=col_labels,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.5)
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#e0e0e0")
    ax.set_title("Per-step time (ms) and speedup", pad=20)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n[Plot] Saved to {path}")
    plt.close()


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 70)
    print("  Benchmark: Numba vs JAX -- BH + Oscillaton")
    print("=" * 70)
    print(f"  T_final      = {T_FINAL}")
    print(f"  Resolutions  = {RESOLUTIONS}")
    print(f"  BH r_max     = {BH_R_MAX}")
    print(f"  Osc r_max    = {OSC_R_MAX_BASE} (adjusted for high N)")
    print()

    bh_numba_res = []
    bh_jax_res = []
    osc_numba_res = []
    osc_jax_res = []

    for N in RESOLUTIONS:
        print(f"\n{'=' * 70}")
        print(f"  N = {N}")
        print(f"{'=' * 70}")

        # -- BH --
        print(f"\n  --- BH (r_max={BH_R_MAX}) ---")
        grid, bg, matter, sv, state = setup_bh(N)
        dr_val = float(grid.min_dr)
        dt_val = 0.4 * dr_val
        n_steps = int(np.ceil(T_FINAL / dt_val))
        print(f"  dr={dr_val:.6f}, dt={dt_val:.6f}, steps={n_steps}")

        nr = bench_numba("BH", grid, bg, matter, state)
        bh_numba_res.append(nr)

        jr = bench_jax("BH", grid, bg, matter, sv, state)
        bh_jax_res.append(jr)

        su = nr["per_step_ms"] / jr["per_step_ms"]
        print(f"\n  >>> BH N={N}: JAX is {su:.1f}x faster per step")

        # -- Oscillaton --
        osc_rmax = max(OSC_R_MAX_BASE, OSC_DR_MIN * N)
        print(f"\n  --- Oscillaton (r_max={osc_rmax:.0f}) ---")
        grid, bg, matter, sv, state, _ = setup_oscillaton(N)
        dr_val = float(grid.min_dr)
        dt_val = 0.4 * dr_val
        n_steps = int(np.ceil(T_FINAL / dt_val))
        print(f"  dr={dr_val:.6f}, dt={dt_val:.6f}, steps={n_steps}")

        nr = bench_numba("Osc", grid, bg, matter, state)
        osc_numba_res.append(nr)

        jr = bench_jax("Osc", grid, bg, matter, sv, state)
        osc_jax_res.append(jr)

        su = nr["per_step_ms"] / jr["per_step_ms"]
        print(f"\n  >>> Osc N={N}: JAX is {su:.1f}x faster per step")

    # =========================================================================
    # Console summary
    # =========================================================================
    print(f"\n{'=' * 70}")
    print("  SUMMARY -- Per-step time (ms) and speedup")
    print(f"{'=' * 70}")
    hdr = (f"  {'N':>6} | {'BH Numba':>10} | {'BH JAX':>8} | "
           f"{'BH Spdup':>9} | {'Osc Numba':>10} | {'Osc JAX':>8} | "
           f"{'Osc Spdup':>9}")
    sep = (f"  {'-' * 6}-+-{'-' * 10}-+-{'-' * 8}-+-"
           f"{'-' * 9}-+-{'-' * 10}-+-{'-' * 8}-+-{'-' * 9}")
    print(hdr)
    print(sep)
    for bn, bj, on, oj in zip(bh_numba_res, bh_jax_res,
                               osc_numba_res, osc_jax_res):
        bs = bn["per_step_ms"] / bj["per_step_ms"]
        os_ = on["per_step_ms"] / oj["per_step_ms"]
        print(f"  {bn['N']:>6} | {bn['per_step_ms']:>9.1f}ms | "
              f"{bj['per_step_ms']:>7.1f}ms | {bs:>8.1f}x | "
              f"{on['per_step_ms']:>9.1f}ms | {oj['per_step_ms']:>7.1f}ms | "
              f"{os_:>8.1f}x")

    print(f"\n{'=' * 70}")
    print("  SUMMARY -- Total evolution time (s) and JIT overhead")
    print(f"{'=' * 70}")
    hdr2 = (f"  {'N':>6} | {'BH Numba':>10} | {'BH JAX':>8} | "
            f"{'BH JIT':>7} | {'Osc Numba':>10} | {'Osc JAX':>8} | "
            f"{'Osc JIT':>8}")
    sep2 = (f"  {'-' * 6}-+-{'-' * 10}-+-{'-' * 8}-+-"
            f"{'-' * 7}-+-{'-' * 10}-+-{'-' * 8}-+-{'-' * 8}")
    print(hdr2)
    print(sep2)
    for bn, bj, on, oj in zip(bh_numba_res, bh_jax_res,
                               osc_numba_res, osc_jax_res):
        print(f"  {bn['N']:>6} | {bn['total_s']:>9.1f}s | "
              f"{bj['total_s']:>7.1f}s | {bj['jit_s']:>6.1f}s | "
              f"{on['total_s']:>9.1f}s | {oj['total_s']:>7.1f}s | "
              f"{oj['jit_s']:>7.1f}s")

    print(f"\n{'=' * 70}")

    # =========================================================================
    # Plot
    # =========================================================================
    plot_path = os.path.join(script_dir, "benchmark_bh_oscillaton.png")
    plot_results(bh_numba_res, bh_jax_res, osc_numba_res, osc_jax_res,
                 plot_path)


if __name__ == "__main__":
    main()

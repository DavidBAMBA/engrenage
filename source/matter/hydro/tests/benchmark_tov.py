#!/usr/bin/env python3
"""
Benchmark: Numba vs JAX — TOV Star Evolution (Cowling & Dynamic)

Runs TOVEvolution.py as subprocess with different configurations
(backend, evolution mode, resolution) and collects timing results.

Usage:
    python examples/TOV/benchmark_tov.py

Environment variables:
    T_FINAL:       Evolution time (default: 2.0)
    RESOLUTIONS:   Comma-separated grid sizes (default: "500,1000,8000")
"""

import os
import sys
import re
import time
import subprocess

# =============================================================================
# Configuration
# =============================================================================
T_FINAL = os.environ.get("T_FINAL", "2.0")
RESOLUTIONS = [int(x) for x in
               os.environ.get("RESOLUTIONS", "500,1000,4000").split(",")]

script_dir = os.path.dirname(os.path.abspath(__file__))
TOV_SCRIPT = os.path.join(script_dir, "TOVEvolution.py")


# =============================================================================
# Run a single configuration
# =============================================================================
def run_tov(N, mode, jax_run):
    """
    Run TOVEvolution.py with given configuration.
    Returns dict with timing results, or None on failure.
    """
    backend = "JAX" if jax_run else "Numba"
    tag = f"{mode}/{backend}/N={N}"
    print(f"\n  [{tag}] Starting...", flush=True)

    env = os.environ.copy()
    env["NUM_POINTS"] = str(N)
    env["EVOLUTION_MODE"] = mode
    env["JAX_RUN"] = "1" if jax_run else "0"
    env["T_FINAL"] = T_FINAL
    env["ENABLE_DATA_SAVING"] = "0"
    env["SKIP_PLOTS"] = "1"
    env["ENABLE_RESTART"] = "0"

    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, TOV_SCRIPT],
            env=env,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout
        )
    except subprocess.TimeoutExpired:
        print(f"  [{tag}] TIMEOUT after 2 hours")
        return None

    wall_time = time.time() - t0

    # Print last lines of output for progress visibility
    stdout_lines = result.stdout.strip().split("\n") if result.stdout else []
    for line in stdout_lines[-5:]:
        print(f"    {line}")

    if result.returncode != 0:
        print(f"  [{tag}] FAILED (exit code {result.returncode})")
        stderr_lines = result.stderr.strip().split("\n") if result.stderr else []
        for line in stderr_lines[-10:]:
            print(f"    STDERR: {line}")
        return None

    # Parse BENCHMARK_RESULT line
    bench_line = None
    for line in stdout_lines:
        if line.startswith("BENCHMARK_RESULT:"):
            bench_line = line
            break

    if bench_line is None:
        print(f"  [{tag}] WARNING: No BENCHMARK_RESULT found in output")
        return None

    # Parse: BENCHMARK_RESULT: jit_s=X per_step_ms=X total_s=X n_steps=X
    parts = {}
    for token in bench_line.replace("BENCHMARK_RESULT:", "").strip().split():
        key, val = token.split("=")
        parts[key] = float(val)

    info = dict(
        N=N,
        backend=backend,
        mode=mode,
        jit_s=parts.get("jit_s", 0.0),
        per_step_ms=parts.get("per_step_ms", 0.0),
        total_s=parts.get("total_s", 0.0),
        n_steps=int(parts.get("n_steps", 0)),
        wall_s=wall_time,
    )

    print(f"  [{tag}] done — {info['per_step_ms']:.1f} ms/step, "
          f"total={info['total_s']:.1f}s, JIT={info['jit_s']:.1f}s, "
          f"wall={wall_time:.1f}s")
    return info


# =============================================================================
# Plotting
# =============================================================================
def plot_results(cow_numba, cow_jax, dyn_numba, dyn_jax, path):
    """Generate 2x2 benchmark comparison plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Ns = [r["N"] for r in cow_numba]
    cn = [r["per_step_ms"] for r in cow_numba]
    cj = [r["per_step_ms"] for r in cow_jax]
    dn = [r["per_step_ms"] for r in dyn_numba]
    dj = [r["per_step_ms"] for r in dyn_jax]

    cow_spdup = [n / j if j > 0 else 0 for n, j in zip(cn, cj)]
    dyn_spdup = [n / j if j > 0 else 0 for n, j in zip(dn, dj)]

    ct_n = [r["total_s"] for r in cow_numba]
    ct_j = [r["total_s"] for r in cow_jax]
    dt_n = [r["total_s"] for r in dyn_numba]
    dt_j = [r["total_s"] for r in dyn_jax]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Benchmark: Numba vs JAX — TOV Star (T={T_FINAL})",
                 fontsize=14, fontweight="bold")

    # -- (0,0) Per-step time --
    ax = axes[0, 0]
    ax.plot(Ns, cn, "o-",  color="tab:blue",  lw=2, ms=7, label="Cow Numba")
    ax.plot(Ns, cj, "s-",  color="tab:orange", lw=2, ms=7, label="Cow JAX")
    ax.plot(Ns, dn, "o--", color="tab:blue",  lw=1.5, ms=5, alpha=0.6, label="Dyn Numba")
    ax.plot(Ns, dj, "s--", color="tab:orange", lw=1.5, ms=5, alpha=0.6, label="Dyn JAX")
    ax.set_xlabel("Grid points N")
    ax.set_ylabel("ms / RK4 step")
    ax.set_title("Per-step time")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    # -- (0,1) Total evolution time --
    ax = axes[0, 1]
    ax.plot(Ns, ct_n, "o-",  color="tab:blue",  lw=2, ms=7, label="Cow Numba")
    ax.plot(Ns, ct_j, "s-",  color="tab:orange", lw=2, ms=7, label="Cow JAX")
    ax.plot(Ns, dt_n, "o--", color="tab:blue",  lw=1.5, ms=5, alpha=0.6, label="Dyn Numba")
    ax.plot(Ns, dt_j, "s--", color="tab:orange", lw=1.5, ms=5, alpha=0.6, label="Dyn JAX")
    ax.set_xlabel("Grid points N")
    ax.set_ylabel("Total time (s)")
    ax.set_title(f"Total evolution (T={T_FINAL})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    # -- (1,0) Speedup --
    ax = axes[1, 0]
    ax.plot(Ns, cow_spdup, "D-", color="tab:green",  lw=2, ms=8, label="Cowling speedup")
    ax.plot(Ns, dyn_spdup, "^-", color="tab:purple", lw=2, ms=8, label="Dynamic speedup")
    ax.axhline(1, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Grid points N")
    ax.set_ylabel("Speedup (Numba / JAX)")
    ax.set_title("Per-step speedup")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    for i, n in enumerate(Ns):
        ax.annotate(f"{cow_spdup[i]:.1f}x", (n, cow_spdup[i]),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9, color="tab:green")
        ax.annotate(f"{dyn_spdup[i]:.1f}x", (n, dyn_spdup[i]),
                    textcoords="offset points", xytext=(0, -15),
                    ha="center", fontsize=9, color="tab:purple")

    # -- (1,1) Summary table --
    ax = axes[1, 1]
    ax.axis("off")
    rows = []
    for i in range(len(Ns)):
        rows.append([
            f"{Ns[i]}",
            f"{cn[i]:.1f}", f"{cj[i]:.1f}", f"{cow_spdup[i]:.1f}x",
            f"{dn[i]:.1f}", f"{dj[i]:.1f}", f"{dyn_spdup[i]:.1f}x",
        ])
    col_labels = ["N",
                  "Cow\nNumba", "Cow\nJAX", "Cow\nSpdup",
                  "Dyn\nNumba", "Dyn\nJAX", "Dyn\nSpdup"]
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
    print("  Benchmark: Numba vs JAX — TOV Star (Cowling & Dynamic)")
    print("=" * 70)
    print(f"  T_final      = {T_FINAL}")
    print(f"  Resolutions  = {RESOLUTIONS}")
    print(f"  Script       = {TOV_SCRIPT}")
    print()

    cow_numba = []
    cow_jax = []
    dyn_numba = []
    dyn_jax = []

    for N in RESOLUTIONS:
        print(f"\n{'=' * 70}")
        print(f"  N = {N}")
        print(f"{'=' * 70}")

        # --- Cowling ---
        print(f"\n  --- Cowling ---")
        r = run_tov(N, "cowling", jax_run=False)
        if r: cow_numba.append(r)

        r = run_tov(N, "cowling", jax_run=True)
        if r: cow_jax.append(r)

        if cow_numba and cow_jax and cow_numba[-1]["N"] == N:
            su = cow_numba[-1]["per_step_ms"] / cow_jax[-1]["per_step_ms"]
            print(f"\n  >>> Cowling N={N}: JAX is {su:.1f}x faster per step")

        # --- Dynamic ---
        print(f"\n  --- Dynamic ---")
        r = run_tov(N, "dynamic", jax_run=False)
        if r: dyn_numba.append(r)

        r = run_tov(N, "dynamic", jax_run=True)
        if r: dyn_jax.append(r)

        if dyn_numba and dyn_jax and dyn_numba[-1]["N"] == N:
            su = dyn_numba[-1]["per_step_ms"] / dyn_jax[-1]["per_step_ms"]
            print(f"\n  >>> Dynamic N={N}: JAX is {su:.1f}x faster per step")

    # =========================================================================
    # Console summary
    # =========================================================================
    if len(cow_numba) == len(RESOLUTIONS) and len(cow_jax) == len(RESOLUTIONS):
        print(f"\n{'=' * 70}")
        print("  SUMMARY — Per-step time (ms) and speedup")
        print(f"{'=' * 70}")
        hdr = (f"  {'N':>6} | {'Cow Numba':>10} | {'Cow JAX':>8} | "
               f"{'Cow Spd':>8} | {'Dyn Numba':>10} | {'Dyn JAX':>8} | "
               f"{'Dyn Spd':>8}")
        sep = (f"  {'-'*6}-+-{'-'*10}-+-{'-'*8}-+-"
               f"{'-'*8}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}")
        print(hdr)
        print(sep)
        for cn_r, cj_r, dn_r, dj_r in zip(cow_numba, cow_jax, dyn_numba, dyn_jax):
            cs = cn_r["per_step_ms"] / cj_r["per_step_ms"] if cj_r["per_step_ms"] > 0 else 0
            ds = dn_r["per_step_ms"] / dj_r["per_step_ms"] if dj_r["per_step_ms"] > 0 else 0
            print(f"  {cn_r['N']:>6} | {cn_r['per_step_ms']:>9.1f}ms | "
                  f"{cj_r['per_step_ms']:>7.1f}ms | {cs:>7.1f}x | "
                  f"{dn_r['per_step_ms']:>9.1f}ms | {dj_r['per_step_ms']:>7.1f}ms | "
                  f"{ds:>7.1f}x")

        print(f"\n{'=' * 70}")
        print("  SUMMARY — Total evolution time (s) and JIT overhead")
        print(f"{'=' * 70}")
        hdr2 = (f"  {'N':>6} | {'Cow Numba':>10} | {'Cow JAX':>8} | "
                f"{'Cow JIT':>8} | {'Dyn Numba':>10} | {'Dyn JAX':>8} | "
                f"{'Dyn JIT':>8}")
        sep2 = (f"  {'-'*6}-+-{'-'*10}-+-{'-'*8}-+-"
                f"{'-'*8}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}")
        print(hdr2)
        print(sep2)
        for cn_r, cj_r, dn_r, dj_r in zip(cow_numba, cow_jax, dyn_numba, dyn_jax):
            print(f"  {cn_r['N']:>6} | {cn_r['total_s']:>9.1f}s | "
                  f"{cj_r['total_s']:>7.1f}s | {cj_r['jit_s']:>7.1f}s | "
                  f"{dn_r['total_s']:>9.1f}s | {dj_r['total_s']:>7.1f}s | "
                  f"{dj_r['jit_s']:>7.1f}s")

        print(f"\n{'=' * 70}")

        # Plot
        plot_path = os.path.join(script_dir, "benchmark_tov.png")
        plot_results(cow_numba, cow_jax, dyn_numba, dyn_jax, plot_path)
    else:
        print("\n  WARNING: Some runs failed, skipping summary and plot.")
        print(f"  Cowling Numba: {len(cow_numba)}/{len(RESOLUTIONS)}")
        print(f"  Cowling JAX:   {len(cow_jax)}/{len(RESOLUTIONS)}")
        print(f"  Dynamic Numba: {len(dyn_numba)}/{len(RESOLUTIONS)}")
        print(f"  Dynamic JAX:   {len(dyn_jax)}/{len(RESOLUTIONS)}")


if __name__ == "__main__":
    main()

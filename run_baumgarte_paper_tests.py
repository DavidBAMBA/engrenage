#!/usr/bin/env python3
"""
Script to replicate figures from Baumgarte, Hughes & Shapiro (1999)
"Evolving Einstein's Field Equations with Matter: The Hydro without Hydro Test"

Generates:
- Figure 1: φ and K evolution at center
- Figure 2: Convergence test (optional, commented out by default - takes longer)
- Figure 3: Outer boundary analysis (optional, commented out by default)
"""

import sys
from pathlib import Path

# Add repo to path
REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))

from source.matter.hydro.tests.test_hydro_w_hydro import (
    run_hwh_test,
    plot_baumgarte_paper_figures,
    plot_baumgarte_fig2_convergence,
    plot_baumgarte_fig3_outer_boundary
)

if __name__ == "__main__":
    print("="*70)
    print("BAUMGARTE PAPER FIGURE REPLICATION")
    print("Paper: gr-qc/9902024v1 (1999)")
    print("="*70)

    # ========== FIGURE 1: φ and K evolution ==========
    print("\n" + "="*70)
    print("FIGURE 1: Long-term evolution of φ and K at center")
    print("="*70)
    print("\nRunning simulation...")
    print("  Grid: ~(32)³")
    print("  EOS: Polytropic n=1 (γ=2)")
    print("  ρ_central: 0.2 (Baumgarte paper)")
    print("  Evolution time: ~100 M")

    result = run_hwh_test(
        gamma=2.0,
        K=1.0,
        rho_central=0.2,  # Baumgarte paper value (M≈0.157, R≈0.866)
        cfl=0.1,
        dr=0.02,  # ~32³ grid for R≈1.6
        t_final_factor=100,  # Evolve to t ≈ 100M
        progress=True,
        save_interval=10
    )

    plot_baumgarte_paper_figures(result, save_plots=True)

    print("\n Figure 1 generated: hwh_plots/baumgarte_fig1_phi_K_evolution.png")

    # ========== FIGURE 2: Convergence test (OPTIONAL - commented out) ==========
    # Uncomment to generate Figure 2 (takes longer - 3 simulations)
    """
    print("\n" + "="*70)
    print("FIGURE 2: Convergence test for K at center")
    print("="*70)

    resolutions = [16, 32, 64]
    results_conv = []

    for Nr_grid in resolutions:
        print(f"\n  Running {Nr_grid}³ grid...")
        R_approx = 1.6
        dr = R_approx / Nr_grid

        res = run_hwh_test(
            gamma=2.0,
            K=1.0,
            rho_central=0.2,
            cfl=0.1,
            dr=dr,
            t_final_factor=20,  # Short evolution to t≈3 for convergence test
            progress=False,
            save_interval=1
        )
        results_conv.append(res)

    plot_baumgarte_fig2_convergence(results_conv, resolutions, save_plots=True)
    print("\n Figure 2 generated: hwh_plots/baumgarte_fig2_convergence.png")
    """

    # ========== FIGURE 3: Outer boundary analysis (OPTIONAL - commented out) ==========
    # Uncomment to generate Figure 3 (takes longer - 3 simulations)
    """
    print("\n" + "="*70)
    print("FIGURE 3: Outer boundary analysis")
    print("="*70)

    ob_locations = [1.0, 2.0, 4.0]
    results_ob = []

    for ob in ob_locations:
        print(f"\n  Running with OB at r={ob}...")

        # Keep resolution constant
        Nr_fixed = 32
        r_max = ob
        dr = r_max / Nr_fixed

        res = run_hwh_test(
            gamma=2.0,
            K=1.0,
            rho_central=0.2,
            cfl=0.1,
            dr=dr,
            r_max=ob,
            t_final_factor=50,  # t ≈ 8M
            progress=False,
            save_interval=1
        )
        results_ob.append(res)

    plot_baumgarte_fig3_outer_boundary(results_ob, ob_locations, save_plots=True)
    print("\n Figure 3 generated: hwh_plots/baumgarte_fig3_outer_boundary.png")
    """

    # ========== Summary ==========
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nTOV star properties:")
    print(f"  Mass:   M = {result['TOV_Mass']:.6f}")
    print(f"  Radius: R = {result['TOV_Radius']:.6f}")
    print(f"  M/R   = {result['TOV_Mass']/result['TOV_Radius']:.4f}")

    print(f"\nEvolution results:")
    print(f"  Final time: t/M = {result['time']/result['TOV_Mass']:.1f}")
    print(f"  Max |K|:        {max(abs(result['K_center'])):.2e}")
    print(f"  φ drift:        {result['phi_center'][-1] - result['phi_center'][0]:.2e}")

    print("\n" + "="*70)
    print("FIGURES GENERATED")
    print("="*70)
    print("\n Figure 1: hwh_plots/baumgarte_fig1_phi_K_evolution.png")
    print("   - Shows φ and K evolution at center")
    print("   - Demonstrates stable long-term evolution")

    print("\nTo generate Figure 2 (convergence test):")
    print("  - Uncomment the Figure 2 section in this script")
    print("  - Takes ~3x longer (runs 3 simulations)")

    print("\nTo generate Figure 3 (outer boundary analysis):")
    print("  - Uncomment the Figure 3 section in this script")
    print("  - Takes ~3x longer (runs 3 simulations)")

    print("\nSee hwh_plots/ directory for all generated figures.")
    print("="*70)

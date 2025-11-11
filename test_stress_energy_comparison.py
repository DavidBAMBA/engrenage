#!/usr/bin/env python3
"""
Test script to compare stress-energy tensor projections:
Method 1 (engrenage): Valencia formulas
Method 2 (NRPy+-like): Direct projection from T^{μν}

This tests if both methods give the same ρ, S_i, S_{ij}, S
"""

import numpy as np
import sys

# Add repo to path
sys.path.insert(0, '/home/yo/repositories/engrenage')

from source.matter.hydro.geometry import ADMGeometry
from source.matter.hydro.stress_energy import StressEnergyTensor

def compute_stress_energy_nrpy_style(geometry: ADMGeometry, rho0, v_U, pressure, W, h):
    """
    Compute stress-energy projections using NRPy+-like method.

    Direct projection from T^{μν}:
    - ρ = n_μ n_ν T^{μν}
    - S_i = -γ_{iμ} n_ν T^{μν}
    - S_{ij} = γ_{iμ} γ_{jν} T^{μν}

    Where:
    - γ_{μν} = g_{μν} + n_μ n_ν  (spatial projector)
    - n^μ = (1/α, -β^i/α)        (unit normal, contravariant)
    - n_μ = (-α, 0, 0, 0)          (unit normal, covariant)
    """
    N = len(rho0)

    # Get 4-metric
    g4UU = geometry.get_4metric_contravariant()
    g4DD = geometry.get_4metric_covariant()

    # Compute T^{μν} for perfect fluid
    # T^{μν} = ρ₀ h u^μ u^ν + P g^{μν}
    u4U = geometry.compute_4velocity(v_U, W)
    rho_h = rho0 * h

    T4UU = np.zeros((N, 4, 4))
    for mu in range(4):
        for nu in range(4):
            T4UU[:, mu, nu] = rho_h * u4U[:, mu] * u4U[:, nu] + pressure * g4UU[:, mu, nu]

    # Covariant unit normal: n_μ = (-α, 0, 0, 0)
    n4D = np.zeros((N, 4))
    n4D[:, 0] = -geometry.alpha

    # Spatial projector: γ_{μν} = g_{μν} + n_μ n_ν
    gamma4DD = np.zeros((N, 4, 4))
    for mu in range(4):
        for nu in range(4):
            gamma4DD[:, mu, nu] = g4DD[:, mu, nu] + n4D[:, mu] * n4D[:, nu]

    # Project to get ADM quantities
    # ρ = n_μ n_ν T^{μν}
    rho_adm = np.zeros(N)
    for mu in range(4):
        for nu in range(4):
            rho_adm += n4D[:, mu] * n4D[:, nu] * T4UU[:, mu, nu]

    # S_i = -γ_{iμ} n_ν T^{μν}  (i=1,2,3 in 4D indexing -> i=0,1,2 in 3D)
    S_D_adm = np.zeros((N, 3))
    for i in range(3):
        for mu in range(4):
            for nu in range(4):
                S_D_adm[:, i] += -gamma4DD[:, i+1, mu] * n4D[:, nu] * T4UU[:, mu, nu]

    # S_{ij} = γ_{iμ} γ_{jν} T^{μν}
    S_DD_adm = np.zeros((N, 3, 3))
    for i in range(3):
        for j in range(3):
            for mu in range(4):
                for nu in range(4):
                    S_DD_adm[:, i, j] += gamma4DD[:, i+1, mu] * gamma4DD[:, j+1, nu] * T4UU[:, mu, nu]

    # S = γ^{ij} S_{ij}
    S_trace = np.einsum('nij,nij->n', geometry.gamma_UU, S_DD_adm)

    return rho_adm, S_D_adm, S_DD_adm, S_trace


def main():
    print("="*70)
    print("COMPARING STRESS-ENERGY TENSOR PROJECTION METHODS")
    print("="*70)

    # Create test data: simple TOV-like configuration
    N = 5

    # Metric: slightly non-trivial (like TOV exterior)
    M = 0.157  # TOV mass
    r = np.linspace(1.0, 2.0, N)

    alpha = np.sqrt(1.0 - 2.0*M/r)
    beta_U = np.zeros((N, 3))
    gamma_LL = np.zeros((N, 3, 3))
    gamma_LL[:, 0, 0] = 1.0 / (1.0 - 2.0*M/r)  # γ_rr
    gamma_LL[:, 1, 1] = r**2                     # γ_θθ
    gamma_LL[:, 2, 2] = r**2                     # γ_φφ

    geometry = ADMGeometry(alpha=alpha, beta_U=beta_U, gamma_LL=gamma_LL)

    # Fluid variables: static (v=0) with modest pressure
    rho0 = np.full(N, 0.05)
    pressure = np.full(N, 0.001)
    v_U = np.zeros((N, 3))  # Static fluid
    W = np.ones(N)           # Lorentz factor = 1
    Gamma = 2.0
    eps = pressure / (rho0 * (Gamma - 1.0))
    h = 1.0 + eps + pressure/rho0

    print(f"\nTest setup:")
    print(f"  N points: {N}")
    print(f"  r range: [{r[0]:.2f}, {r[-1]:.2f}]")
    print(f"  rho0: {rho0[0]:.3f}")
    print(f"  P: {pressure[0]:.3e}")
    print(f"  v^r: {v_U[0,0]:.3e} (static)")

    # Method 1: Valencia formulas (current engrenage)
    print("\n" + "-"*70)
    print("METHOD 1: Valencia formulas (current engrenage)")
    print("-"*70)
    st = StressEnergyTensor(geometry, rho0, v_U, pressure, W, h)
    em1 = st.project_to_ADM()

    print(f"  ρ:      {em1.rho[2]:.6e}")
    print(f"  S_r:    {em1.S_D[2, 0]:.6e}")
    print(f"  S_rr:   {em1.S_DD[2, 0, 0]:.6e}")
    print(f"  S_θθ:   {em1.S_DD[2, 1, 1]:.6e}")
    print(f"  S:      {em1.S[2]:.6e}")

    # Method 2: Direct projection (NRPy+-like)
    print("\n" + "-"*70)
    print("METHOD 2: Direct projection from T^μν (NRPy+-like)")
    print("-"*70)
    rho2, S_D2, S_DD2, S2 = compute_stress_energy_nrpy_style(
        geometry, rho0, v_U, pressure, W, h)

    print(f"  ρ:      {rho2[2]:.6e}")
    print(f"  S_r:    {S_D2[2, 0]:.6e}")
    print(f"  S_rr:   {S_DD2[2, 0, 0]:.6e}")
    print(f"  S_θθ:   {S_DD2[2, 1, 1]:.6e}")
    print(f"  S:      {S2[2]:.6e}")

    # Compare
    print("\n" + "="*70)
    print("COMPARISON (Method 1 vs Method 2)")
    print("="*70)

    diff_rho = np.abs(em1.rho - rho2)
    diff_S_D = np.abs(em1.S_D - S_D2)
    diff_S_DD = np.abs(em1.S_DD - S_DD2)
    diff_S = np.abs(em1.S - S2)

    print(f"\nMaximum absolute differences:")
    print(f"  |Δρ|:       {np.max(diff_rho):.3e}")
    print(f"  |ΔS_i|:     {np.max(diff_S_D):.3e}")
    print(f"  |ΔS_ij|:    {np.max(diff_S_DD):.3e}")
    print(f"  |ΔS|:       {np.max(diff_S):.3e}")

    # Check if methods agree
    tol = 1e-10
    if (np.max(diff_rho) < tol and np.max(diff_S_D) < tol and
        np.max(diff_S_DD) < tol and np.max(diff_S) < tol):
        print(f"\n✅ PASS: Both methods agree to within tolerance {tol:.0e}")
        return 0
    else:
        print(f"\n❌ FAIL: Methods differ by more than tolerance {tol:.0e}")
        print("\nDetailed comparison at center point:")
        i = 2
        print(f"  ρ:   Method1={em1.rho[i]:.10e}, Method2={rho2[i]:.10e}, diff={diff_rho[i]:.3e}")
        print(f"  S_r: Method1={em1.S_D[i,0]:.10e}, Method2={S_D2[i,0]:.10e}, diff={diff_S_D[i,0]:.3e}")
        print(f"  S:   Method1={em1.S[i]:.10e}, Method2={S2[i]:.10e}, diff={diff_S[i]:.3e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

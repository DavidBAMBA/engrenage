"""
Complete comparison: Valencia vs NRPy GRHD_equations.py

Compares ALL intermediate quantities:
1. Stress-energy tensors T^μν and T^μ_ν
2. Physical fluxes F^j (density, energy, momentum)
3. Connection terms (density, energy, momentum)
4. Source terms (energy and momentum, with sub-terms)

Reference: nrpy/nrpy/equations/grhd/GRHD_equations.py
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

np.random.seed(42)

from source.matter.hydro.tests.grhd_equations_numpy import GRHD_Equations_NumPy
from source.matter.hydro.valencia_reference_metric import ValenciaReferenceMetric
from source.bssn.bssnvars import BSSNVars
from source.bssn.tensoralgebra import get_bar_gamma_LL, get_bar_A_LL, SPACEDIM
from source.backgrounds.sphericalbackground import FlatSphericalBackground, i_r, i_t, i_p
from source.matter.hydro.eos import PolytropicEOS
from source.core.spacing import NUM_GHOSTS

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*80)
    print(title.center(80))
    print("="*80)

def print_subsection(title):
    """Print a formatted subsection header"""
    print("\n" + title)
    print("-" * 80)

def compare_scalar(name, val1, val2, tolerance=1e-12):
    """Compare scalar arrays and print result"""
    diff = np.max(np.abs(val1 - val2))
    status = "✓ PASS" if diff < tolerance else "✗ FAIL"
    print(f"{name:45s}: max diff = {diff:.3e}  {status}")
    return diff < tolerance

def compare_vector(name, val1, val2, tolerance=1e-12):
    """Compare vector arrays and print result with components"""
    diff = np.max(np.abs(val1 - val2))
    status = "✓ PASS" if diff < tolerance else "✗ FAIL"
    print(f"{name:45s}: max diff = {diff:.3e}  {status}")
    for i in range(val1.shape[1]):
        diff_i = np.max(np.abs(val1[:, i] - val2[:, i]))
        print(f"  Component {i}: {diff_i:.3e}")
    return diff < tolerance

def compare_tensor(name, val1, val2, tolerance=1e-12):
    """Compare tensor arrays and print result with components"""
    diff = np.max(np.abs(val1 - val2))
    status = "✓ PASS" if diff < tolerance else "✗ FAIL"
    print(f"{name:45s}: max diff = {diff:.3e}  {status}")
    for i in range(val1.shape[1]):
        for j in range(val1.shape[2]):
            diff_ij = np.max(np.abs(val1[:, i, j] - val2[:, i, j]))
            if diff_ij > tolerance * 0.1:  # Only print non-negligible components
                print(f"  Component ({i},{j}): {diff_ij:.3e}")
    return diff < tolerance

print_section("COMPLETE VALENCIA VS NRPY COMPARISON")

# ============================================================================
# SETUP
# ============================================================================
print_subsection("Setup: Creating test configuration")

N = 100
r_min, r_max = 1.0, 10.0
r = np.linspace(r_min, r_max, N)

background = FlatSphericalBackground(r)
bssn_vars = BSSNVars(N)
bssn_d1 = BSSNVars(N)

# Initialize BSSN variables
bssn_vars.phi = np.random.uniform(-0.1, 0.1, N)
bssn_vars.K = np.random.uniform(-0.1, 0.1, N)
bssn_vars.lapse = np.ones(N) + np.random.uniform(-0.05, 0.05, N)
bssn_vars.shift_U = np.random.uniform(-0.01, 0.01, (N, SPACEDIM))
bssn_vars.h_LL = np.random.uniform(-0.1, 0.1, (N, SPACEDIM, SPACEDIM))
bssn_vars.A_LL = np.random.uniform(-0.1, 0.1, (N, SPACEDIM, SPACEDIM))

# Derivatives
bssn_d1.phi = np.random.uniform(-0.01, 0.01, (N, SPACEDIM))
bssn_d1.lapse = np.random.uniform(-0.01, 0.01, (N, SPACEDIM))
bssn_d1.shift_U = np.random.uniform(-0.001, 0.001, (N, SPACEDIM, SPACEDIM))
bssn_d1.h_LL = np.random.uniform(-0.01, 0.01, (N, SPACEDIM, SPACEDIM, SPACEDIM))

# Hydrodynamic primitives
rho0 = np.ones(N) * 1.0 + np.random.uniform(-0.1, 0.1, N)
pressure = np.ones(N) * 0.1 + np.random.uniform(-0.01, 0.01, N)

# EOS and derived quantities
eos = PolytropicEOS(K=100.0, gamma=2.0)
eps = eos.eps_from_rho_p(rho0, pressure)
h = 1.0 + eps + pressure / rho0

# Lorentz factor and velocities (ensure v² << 1)
e4phi = np.exp(4.0 * bssn_vars.phi)
bar_gamma_LL = get_bar_gamma_LL(r, bssn_vars.h_LL, background)
gamma_LL = e4phi[:, None, None] * bar_gamma_LL

# Generate velocities ensuring v² < 0.01 (v < 0.1c)
v_U = np.zeros((N, SPACEDIM))
for i in range(N):
    # Generate small random velocities
    v_test = np.random.uniform(-0.05, 0.05, SPACEDIM)
    v_squared_test = np.einsum('ij,i,j->', gamma_LL[i], v_test, v_test)
    # Scale down if too large
    if v_squared_test > 0.01:
        v_test *= 0.1 / np.sqrt(v_squared_test)
    v_U[i] = v_test

v_squared = np.einsum('xij,xi,xj->x', gamma_LL, v_U, v_U)
W = 1.0 / np.sqrt(np.maximum(1.0 - v_squared, 1e-16))

print(f"  Grid points: {N}")
print(f"  Radial range: [{r_min}, {r_max}]")
print(f"  Mean density: {np.mean(rho0):.3e}")
print(f"  Mean pressure: {np.mean(pressure):.3e}")
print(f"  Mean Lorentz factor: {np.mean(W):.3f}")

# ============================================================================
# INITIALIZE VALENCIA
# ============================================================================
print_subsection("Initializing Valencia")

valencia = ValenciaReferenceMetric()
valencia._extract_geometry(r, bssn_vars, "dynamic", background, None)

# Compute source terms to populate debug variables
src_S_val, src_tau_val = valencia._compute_source_terms(
    rho0, v_U, pressure, W, h, bssn_vars, bssn_d1, background, "dynamic", r
)

print(f"  Valencia initialized successfully")

# ============================================================================
# INITIALIZE NRPY
# ============================================================================
print_subsection("Initializing NRPy equations")

grhd = GRHD_Equations_NumPy(N)

# Set geometric quantities
alpha_val = valencia.alpha
e6phi_val = np.exp(6.0 * bssn_vars.phi)
beta_U_val = valencia.beta_U
gamma_LL_val = valencia.gamma_LL

grhd.alpha = alpha_val
grhd.e6phi = e6phi_val
grhd.betaU = beta_U_val
grhd.gammaDD = gamma_LL_val

# Compute inverse metric gammaUU
gamma_UU = np.zeros_like(gamma_LL_val)
for i in range(N):
    gamma_UU[i] = np.linalg.inv(gamma_LL_val[i])
grhd.gammaUU = gamma_UU

# Set primitives
grhd.rho_b = rho0
grhd.P = pressure
grhd.h = h
grhd.u4U_0 = W
grhd.VU = v_U

# Compute 4-velocity
u4U_val = valencia._compute_4velocity(rho0, v_U, W)
# u4U should be (N, 4) array with [u^0, u^1, u^2, u^3]
grhd.u4U = u4U_val

print(f"  NRPy initialized successfully")

# ============================================================================
# PART 1: STRESS-ENERGY TENSORS
# ============================================================================
print_section("PART 1: STRESS-ENERGY TENSORS")

# Compute T^μν
grhd.compute_T4UU()
T4UU_val = valencia._debug_T4UU

print_subsection("1.1 Contravariant stress-energy tensor T^μν")
compare_scalar("T^00", T4UU_val['00'], grhd.T4UU['00'])
compare_vector("T^0i", T4UU_val['0i'], grhd.T4UU['0i'])
compare_tensor("T^ij", T4UU_val['ij'], grhd.T4UU['ij'])

# Compute T^μ_ν
grhd.compute_T4UD()
T4UD_val = valencia._debug_T4UD

print_subsection("1.2 Mixed stress-energy tensor T^μ_ν")
compare_scalar("T^0_0", T4UD_val['0_0'], grhd.T4UD['0_0'])
compare_vector("T^0_j", T4UD_val['0_j'], grhd.T4UD['0_j'])
compare_tensor("T^i_j", T4UD_val['i_j'], grhd.T4UD['i_j'])

# ============================================================================
# PART 2: PHYSICAL FLUXES
# ============================================================================
print_section("PART 2: PHYSICAL FLUXES")

# Need to call compute_rhs to populate flux debug variables
dummy_D = alpha_val * e6phi_val * rho0 * W
dummy_tau = alpha_val * e6phi_val * (rho0 * h * W**2 - pressure) - dummy_D
dummy_S = dummy_D[:, np.newaxis] * v_U  # Broadcasting: (N,) × (N,3) → (N,3)

from source.matter.hydro.riemann import HLLRiemannSolver
from source.matter.hydro.reconstruction import Reconstruction

reconstructor = Reconstruction(method="minmod")
riemann_solver = HLLRiemannSolver()

valencia.compute_rhs(
    dummy_D, dummy_S, dummy_tau, rho0, v_U, pressure, W, h,
    r, bssn_vars, bssn_d1, background, "dynamic",
    eos, None, reconstructor, riemann_solver
)

# Compute NRPy fluxes
grhd.compute_rho_star()
grhd.compute_rho_star_fluxU()
grhd.compute_tau_tilde_fluxU()
grhd.compute_S_tildeD()
grhd.compute_S_tilde_fluxUD()

print_subsection("2.1 Density flux F^j_ρ")
compare_vector("F^j_ρ (ρ* v^j)", valencia._debug_flux_density, grhd.rho_star_fluxU)

print_subsection("2.2 Energy flux F^j_τ")
compare_vector("F^j_τ (α² e^6φ T^0j - ρ* v^j)", valencia._debug_flux_energy, grhd.tau_tilde_fluxU)

print_subsection("2.3 Momentum flux F^j_i")
compare_tensor("F^j_i (α e^6φ T^i_j)", valencia._debug_flux_momentum, grhd.S_tilde_fluxUD)

# ============================================================================
# PART 3: CONNECTION TERMS
# ============================================================================
print_section("PART 3: CONNECTION TERMS")

grhd.GammahatUDD = background.hat_christoffel
grhd.compute_rho_star__Ye_star__and_tau_connection_terms()
grhd.compute_S_tilde_connection_termsD()

print_subsection("3.1 Density connection term")
print("  Note: Valencia uses -Γ̂^k_{kj} F^j convention, NRPy uses +Γ̂^k_{kj} F^j")
print("        Comparing with sign flip...")
compare_scalar("Γ̂^k_{kj} F^j_ρ", -valencia._debug_connection_density, grhd.rho_star_connection_term)

print_subsection("3.2 Energy connection term")
print("  Note: Valencia uses -Γ̂^k_{kj} F^j convention, NRPy uses +Γ̂^k_{kj} F^j")
print("        Comparing with sign flip...")
compare_scalar("Γ̂^k_{kj} F^j_τ", -valencia._debug_connection_energy, grhd.tau_connection_term)

print_subsection("3.3 Momentum connection term")
print("  Note: Valencia uses different sign convention for connection terms")
print("        Comparing with sign flip...")
compare_vector("Connection term for S̃_i", -valencia._debug_connection_momentum, grhd.S_tilde_connection_termsD)

# ============================================================================
# PART 4: SOURCE TERMS
# ============================================================================
print_section("PART 4: SOURCE TERMS")

# Energy source
bar_A_LL = get_bar_A_LL(r, bssn_vars, background)
K_LL = e4phi[:, None, None] * bar_A_LL + (bssn_vars.K / 3.0)[:, None, None] * gamma_LL_val

grhd.KDD = K_LL
grhd.alpha_dD = bssn_d1.lapse
grhd.compute_tau_source_term()

print_subsection("4.1 Energy source term")
compare_scalar("S_τ: K_ij contraction term", valencia._debug_energy_source_Kij_term, grhd._tau_source_Kij_term, tolerance=1e-10)
compare_scalar("S_τ: ∂α term", valencia._debug_energy_source_dalpha_term, grhd._tau_source_dalpha_term, tolerance=1e-10)
compare_scalar("S_τ: Total (α e^6φ × sum)", valencia._debug_energy_source_total, grhd.tau_source_term, tolerance=1e-10)

# Momentum source
print_subsection("4.2 Momentum source term")

# Need to set up betaU_dD and gamma derivatives for NRPy
betaU_dD_val = (background.d1_inverse_scaling_vector * bssn_vars.shift_U[:, :, np.newaxis]
                + bssn_d1.shift_U * background.inverse_scaling_vector[:, :, np.newaxis])
betaU_dD_nrpy = betaU_dD_val.transpose(0, 2, 1)  # NRPy convention
grhd.betaU_dD = betaU_dD_nrpy

# Compute gammabarDD and derivatives
gammabarDD = bar_gamma_LL  # This is already the conformal metric γ̄_{ij}
gammabarDD_dD = np.zeros((N, SPACEDIM, SPACEDIM, SPACEDIM))
d1_hat_gamma = np.zeros((N, SPACEDIM, SPACEDIM, SPACEDIM))
d1_hat_gamma[:, i_t, i_t, i_r] = 2.0 * background.r
d1_hat_gamma[:, i_p, i_p, i_r] = 2.0 * background.r

for i in range(SPACEDIM):
    for j in range(SPACEDIM):
        for k in range(SPACEDIM):
            gammabarDD_dD[:, i, j, k] = (
                bssn_d1.h_LL[:, i, j, k] * background.scaling_matrix[:, i, j]
                + bssn_vars.h_LL[:, i, j] * background.d1_scaling_matrix[:, i, j, k]
                + d1_hat_gamma[:, i, j, k]
            )

phi_dD = bssn_d1.phi
grhd.compute_S_tilde_source_termD(gammabarDD, gammabarDD_dD, phi_dD)

# Debug: Check if betaU and derivatives match
print(f"\n  Debug: Comparing β^i and ∇̂_i β^j:")
print(f"    β^i diff: {np.max(np.abs(valencia.beta_U - grhd.betaU)):.6e}")
print(f"    Γ̂ diff: {np.max(np.abs(background.hat_christoffel - grhd.GammahatUDD)):.6e}")

# Check partial derivatives
val_d1_Shift_test = (background.d1_inverse_scaling_vector * bssn_vars.shift_U[:,:,np.newaxis]
                    + bssn_d1.shift_U * background.inverse_scaling_vector[:,:,np.newaxis])
print(f"    ∂_r β^r diff: {np.max(np.abs(val_d1_Shift_test[:, i_r, i_r] - betaU_dD_val[:, i_r, i_r])):.6e}")
print(f"    ∂_r β^θ diff: {np.max(np.abs(val_d1_Shift_test[:, i_t, i_r] - betaU_dD_val[:, i_t, i_r])):.6e}")
print(f"    ∂_r β^φ diff: {np.max(np.abs(val_d1_Shift_test[:, i_p, i_r] - betaU_dD_val[:, i_p, i_r])):.6e}")

# Compute Valencia's ∇̂_i β^j for comparison
val_hatD_beta_test = valencia._debug_hatD_beta_U

# NRPy computes it implicitly in the loop, let's reconstruct it
# Careful: NRPy uses GammahatUDD which we set to background.hat_christoffel
nrpy_hatD_beta_test = np.zeros((N, SPACEDIM, SPACEDIM))
for i in range(SPACEDIM):
    for j in range(SPACEDIM):
        nrpy_hatD_beta_test[:, i, j] = betaU_dD_nrpy[:, j, i]  # ∂_i β^j
        for k in range(SPACEDIM):
            nrpy_hatD_beta_test[:, i, j] += grhd.GammahatUDD[:, j, i, k] * grhd.betaU[:, k]

print(f"    ∇̂_r β^r diff: {np.max(np.abs(val_hatD_beta_test[:, i_r, i_r] - nrpy_hatD_beta_test[:, i_r, i_r])):.6e}")
print(f"    ∇̂_r β^θ diff: {np.max(np.abs(val_hatD_beta_test[:, i_r, i_t] - nrpy_hatD_beta_test[:, i_r, i_t])):.6e}")
print(f"    ∇̂_r β^φ diff: {np.max(np.abs(val_hatD_beta_test[:, i_r, i_p] - nrpy_hatD_beta_test[:, i_r, i_p])):.6e}")
print(f"    Overall ∇̂_i β^j diff: {np.max(np.abs(val_hatD_beta_test - nrpy_hatD_beta_test)):.6e}")

compare_vector("S_{S_i}: -T^00 α ∂_i α term",
               valencia._debug_momentum_source_T00_alpha_term,
               grhd._momentum_source_T00_alpha_term,
               tolerance=1e-10)

compare_vector("S_{S_i}: T^0_j ∇̂_i β^j term",
               valencia._debug_momentum_source_T0j_beta_term,
               grhd._momentum_source_T0j_beta_term,
               tolerance=1e-10)

compare_vector("S_{S_i}: Metric derivative term",
               valencia._debug_momentum_source_metric_term,
               grhd._momentum_source_metric_term,
               tolerance=1e-10)

compare_vector("S_{S_i}: Total (α e^6φ × sum)",
               valencia._debug_momentum_source_total,
               grhd.S_tilde_source_termD,
               tolerance=1e-10)

# ============================================================================
# SUMMARY
# ============================================================================
print_section("COMPARISON SUMMARY")

print("""
All intermediate quantities have been compared:
  ✓ Stress-energy tensors (T^μν and T^μ_ν)
  ✓ Physical fluxes (density, energy, momentum)
  ✓ Connection terms (density, energy, momentum)
  ✓ Source terms (energy and momentum with sub-terms)

Check results above for detailed differences.
""")

print("="*80)

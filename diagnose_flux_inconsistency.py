"""
FASE 1 DIAGNÓSTICO: Verificar inconsistencia en formulación de flujos.

Este script verifica si la diferencia entre:
1. Flujos del Riemann solver (simplificados): F^r_Sr = Sr * v^r + p
2. Flujos de connection terms (tensor completo): F^r_r = α e^6φ T^r_r

es la causa del crecimiento de velocidad en TOV.

Verificaciones:
- ¿Hay shift β^r ≠ 0 durante la evolución?
- ¿Qué magnitud tiene la diferencia entre las dos formulaciones?
- ¿Dónde ocurre la máxima diferencia?
"""

import numpy as np
import sys
import os

# Change to TOV directory
tov_dir = os.path.join(os.path.dirname(__file__), 'examples', 'TOV')
os.chdir(tov_dir)
sys.path.insert(0, tov_dir)

from tov_solver import TOVSolver
import tov_initial_data_interpolated as tov_id

# Add repo root
repo_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, repo_root)

from source.core.statevector import StateVector
from source.core.spacing import LinearSpacing
from source.core.grid import Grid
from source.matter.perfect_fluid import PerfectFluid
from source.matter.hydro.eos import IdealGasEOS
from source.backgrounds.sphericalbackground import FlatSphericalBackground
from source.matter.hydro.atmosphere import AtmosphereParams
from source.bssn.bssnvars import BSSNVars
from source.matter.hydro.cons2prim import Cons2PrimSolver
from source.bssn.tensoralgebra import get_bar_gamma_LL

print("=" * 80)
print("FASE 1 DIAGNÓSTICO: Inconsistencia en Flujos")
print("=" * 80)

# Setup
N = 256
r_max = 20.0
K = 100.0
Gamma = 2.0
rho_central = 1.28e-3

spacing = LinearSpacing(N, r_max)
eos = IdealGasEOS(gamma=Gamma)
hydro = PerfectFluid(eos=eos, spacetime_mode="dynamic")
state_vector = StateVector(hydro)
grid = Grid(spacing, state_vector)
background = FlatSphericalBackground(grid.r)

atmosphere = AtmosphereParams(
    rho_floor=1.0e-10,
    p_floor=1.0e-11,
    v_max=0.9999,
    W_max=100.0,
    tau_atm_factor=1.0,
    conservative_floor_safety=0.999999
)

print(f"\nGrid: N={N}, r_max={r_max:.1f}, dr={spacing.dx:.4f}")

# Solve TOV
print("\nSolving TOV...")
tov_solver = TOVSolver(K=K, Gamma=Gamma)
tov_solution = tov_solver.solve(rho_central, r_max=15.0)

print(f"  M_star = {tov_solution['M_star']:.6f}")
print(f"  R_star = {tov_solution['R']:.4f}")

# Create initial data
print("\nCreating initial data...")
initial_state = tov_id.create_initial_data_interpolated(
    tov_solution, grid, background, eos,
    atmosphere=atmosphere,
    interp_order=11
)

# Extract BSSN and matter
from source.bssn.bssnstatevariables import NUM_BSSN_VARS
bssn_vars = BSSNVars(grid.N)
bssn_vars.set_bssn_vars(initial_state[:NUM_BSSN_VARS, :])
hydro.set_matter_vars(initial_state, bssn_vars, grid)

# Get primitives
bar_gamma = get_bar_gamma_LL(grid.r, bssn_vars.h_LL, background)
em4phi = np.exp(-4.0 * bssn_vars.phi)
e6phi = np.exp(6.0 * bssn_vars.phi)
gamma_rr = em4phi * bar_gamma[:, 0, 0]
alpha = bssn_vars.lapse

# Get shift
beta_r = np.zeros(N)
if hasattr(bssn_vars, 'shift_U') and bssn_vars.shift_U is not None:
    shift_arr = np.asarray(bssn_vars.shift_U)
    if shift_arr.ndim >= 2 and shift_arr.shape[1] > 0:
        beta_r = shift_arr[:, 0]
    elif shift_arr.ndim == 1:
        beta_r = shift_arr

cons2prim = Cons2PrimSolver(eos=eos, atmosphere=atmosphere)
metric = (alpha, beta_r, gamma_rr)
U = (hydro.D, hydro.Sr, hydro.tau)
primitives = cons2prim.convert(U=U, metric=metric, p_guess=None, apply_conservative_floors=False)

rho0 = primitives['rho0']
vr = primitives['vr']
p = primitives['p']
W = primitives['W']
Sr = hydro.Sr

# Ghost cells (standard)
num_ghosts = 3

print("\n" + "=" * 80)
print("FASE 1.1: VERIFICACIÓN DE SHIFT β^r")
print("=" * 80)

max_beta = np.max(np.abs(beta_r))
max_beta_idx = np.argmax(np.abs(beta_r))

print(f"\nShift β^r en t=0:")
print(f"  max(|β^r|) = {max_beta:.6e} at i={max_beta_idx}, r={grid.r[max_beta_idx]:.2f}")
print(f"  mean(|β^r|) = {np.mean(np.abs(beta_r)):.6e}")

if max_beta < 1e-14:
    print("\n✓ Shift es esencialmente CERO (< 1e-14)")
    print("  → La diferencia vtil vs v^r NO debería ser importante")
    shift_matters = False
else:
    print(f"\n⚠️  Shift es NO-CERO: max(|β^r|) = {max_beta:.3e}")
    print("  → La diferencia vtil vs v^r PODRÍA ser importante")
    shift_matters = True

# Profile of shift
print(f"\nPerfil de β^r (primeros 20 puntos):")
print(f"{'i':>4} {'r':>8} {'β^r':>12} {'α':>12}")
print("-" * 40)
for i in range(num_ghosts, min(num_ghosts + 20, N - num_ghosts)):
    print(f"{i:4d} {grid.r[i]:8.2f} {beta_r[i]:+12.3e} {alpha[i]:12.6f}")

print("\n" + "=" * 80)
print("FASE 1.2: COMPARACIÓN DE FORMULACIONES DE FLUJOS")
print("=" * 80)

# Compute fluxes with BOTH formulations

# 1. Riemann formulation (SIMPLIFIED)
# F^r_Sr = Sr * v^r + p
# But Riemann uses Valencia transport velocity: vtil = v^r - β^r/α

vtil = vr - beta_r / alpha  # Valencia transport velocity
F_Sr_riemann = Sr * vtil + p

print(f"\n1. Flujos del Riemann Solver (simplificados):")
print(f"   F^r_Sr = Sr × (v^r - β^r/α) + p")
print(f"   where vtil = v^r - β^r/α (velocidad de transporte Valencia)")

# 2. Connection terms formulation (COMPLETE)
# F^r_r = α e^{6φ} T^r_r
# T^r_r = ρ₀ h W² (v^r)² + P (γ^{rr} - β^r β^r / α²)

# Compute specific enthalpy
# For ideal gas: eps = p / ((Gamma-1) * rho0)
# h = 1 + eps + p/rho0 = 1 + Gamma * p / ((Gamma-1) * rho0)
eps = p / ((Gamma - 1.0) * rho0)
h = 1.0 + eps + p / rho0

# Compute metric components
gamma_UU_rr = 1.0 / gamma_rr  # γ^{rr}
g_UU_rr = gamma_UU_rr - beta_r * beta_r / (alpha * alpha)  # g^{rr} = γ^{rr} - β^r β^r / α²

# Compute stress tensor component T^r_r
Trr = rho0 * h * W * W * vr * vr + p * g_UU_rr

# Compute flux
F_Sr_connection = alpha * e6phi * Trr

print(f"\n2. Flujos de Connection Terms (tensor completo):")
print(f"   F^r_r = α × e^{{6φ}} × T^r_r")
print(f"   T^r_r = ρ₀ h W² (v^r)² + P (γ^{{rr}} - β^r β^r / α²)")

# Compute difference
flux_diff = F_Sr_connection - F_Sr_riemann
max_diff = np.max(np.abs(flux_diff))
max_diff_idx = np.argmax(np.abs(flux_diff))

# Relative difference (where Sr is significant)
significant_mask = np.abs(Sr) > 1e-15
if np.any(significant_mask):
    rel_diff = np.abs(flux_diff[significant_mask]) / (np.abs(F_Sr_riemann[significant_mask]) + 1e-30)
    max_rel_diff = np.max(rel_diff)
else:
    max_rel_diff = 0.0

print(f"\n3. Diferencia entre formulaciones:")
print(f"   ΔF = F_connection - F_riemann")
print(f"   max(|ΔF|) = {max_diff:.6e} at i={max_diff_idx}, r={grid.r[max_diff_idx]:.2f}")
if np.any(significant_mask):
    print(f"   max(|ΔF/F_riemann|) = {max_rel_diff:.6e} (donde |Sr| > 1e-15)")

# Find stellar surface
threshold = 0.01 * rho0[num_ghosts]
surface_idx = None
for i in range(num_ghosts, N - num_ghosts):
    if rho0[i] < threshold:
        surface_idx = i
        break

if surface_idx:
    r_surface = grid.r[surface_idx]
    print(f"\nEn la superficie estelar (r={r_surface:.2f}):")
    print(f"   F_riemann = {F_Sr_riemann[surface_idx]:+.6e}")
    print(f"   F_connection = {F_Sr_connection[surface_idx]:+.6e}")
    print(f"   Diferencia = {flux_diff[surface_idx]:+.6e}")

# Detailed comparison at key locations
print(f"\n" + "=" * 80)
print("COMPARACIÓN DETALLADA EN UBICACIONES CLAVE")
print("=" * 80)

locations = [
    ("Centro", num_ghosts),
    ("Interior", num_ghosts + 30),
    ("Superficie", surface_idx if surface_idx else N//2),
    ("Atmósfera", surface_idx + 10 if surface_idx else N//2 + 10)
]

print(f"\n{'Ubicación':<12} {'r':>8} {'Sr':>12} {'F_riemann':>14} {'F_connection':>14} {'ΔF':>14} {'β^r':>12}")
print("-" * 100)
for name, idx in locations:
    if idx < N - num_ghosts:
        print(f"{name:<12} {grid.r[idx]:8.2f} {Sr[idx]:+12.3e} {F_Sr_riemann[idx]:+14.6e} "
              f"{F_Sr_connection[idx]:+14.6e} {flux_diff[idx]:+14.6e} {beta_r[idx]:+12.3e}")

print(f"\n" + "=" * 80)
print("ANÁLISIS DE TÉRMINOS")
print("=" * 80)

# Break down the difference
print(f"\nDesglose de T^r_r:")
term1 = rho0 * h * W * W * vr * vr  # Kinetic term
term2 = p * gamma_UU_rr  # Pressure with γ^{rr}
term3 = -p * beta_r * beta_r / (alpha * alpha)  # Pressure correction from shift

print(f"En la superficie (r={grid.r[surface_idx]:.2f}):")
print(f"  T^r_r = (ρ₀ h W² v² term) + (P γ^{{rr}} term) + (P shift correction)")
print(f"        = {term1[surface_idx]:+.6e} + {term2[surface_idx]:+.6e} + {term3[surface_idx]:+.6e}")
print(f"        = {Trr[surface_idx]:+.6e}")
print(f"\n  β^r correction to pressure: {term3[surface_idx]:+.6e}")
print(f"  Relative size: {abs(term3[surface_idx]/term2[surface_idx]) if term2[surface_idx] != 0 else 0:.6e}")

# Valencia transport velocity vs regular velocity
print(f"\nDiferencia en velocidades:")
print(f"  v^r = {vr[surface_idx]:+.6e}")
print(f"  vtil = v^r - β^r/α = {vtil[surface_idx]:+.6e}")
print(f"  Diferencia: {vtil[surface_idx] - vr[surface_idx]:+.6e}")

print(f"\n" + "=" * 80)
print("VEREDICTO")
print("=" * 80)

if not shift_matters and max_rel_diff < 1e-10:
    print("\n✓ Las formulaciones son esencialmente EQUIVALENTES:")
    print(f"  - Shift β^r ≈ 0 (max = {max_beta:.3e})")
    print(f"  - Diferencia relativa < 1e-10")
    print("\n→ Esta inconsistencia probablemente NO es la causa del problema")
    print("→ Buscar causa en otro lugar (condiciones iniciales, ecuación de energía, etc.)")
    verdict = "NO_ISSUE"
elif shift_matters or max_rel_diff > 1e-6:
    print("\n⚠️  Las formulaciones son DIFERENTES:")
    if shift_matters:
        print(f"  - Shift β^r ≠ 0 (max = {max_beta:.3e})")
    print(f"  - Diferencia relativa máxima: {max_rel_diff:.3e}")
    print("\n→ Esta inconsistencia PODRÍA ser la causa del problema")
    print("→ Proceder con FASE 2: Implementar corrección unificada")
    verdict = "POTENTIAL_ISSUE"
else:
    print("\n⚠️  Situación no clara:")
    print(f"  - Shift: {max_beta:.3e}")
    print(f"  - Diferencia relativa: {max_rel_diff:.3e}")
    print("\n→ Requiere más investigación")
    verdict = "UNCLEAR"

# Save summary
summary_file = "flux_inconsistency_diagnosis.txt"
with open(summary_file, 'w') as f:
    f.write("DIAGNÓSTICO DE INCONSISTENCIA EN FLUJOS\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"max(|β^r|) = {max_beta:.6e}\n")
    f.write(f"max(|ΔF|) = {max_diff:.6e}\n")
    f.write(f"max(|ΔF/F|) = {max_rel_diff:.6e}\n")
    f.write(f"\nVeredicto: {verdict}\n")

print(f"\nResumen guardado en: {summary_file}")

#!/usr/bin/env python3
"""
Quick test to check if TOV solver works with isotropic coordinates
"""

import sys
sys.path.append('/home/yo/repositories/engrenage')

from examples.tov_solver import TOVSolver
import time

print("Testing TOV solver with isotropic coordinates...")
print("=" * 60)

# Parameters matching hydro_without_hydro_test
K = 100.0
Gamma = 2.0
rho_central = 1.28e-3

print(f"Parameters: K={K}, Gamma={Gamma}, rho_c={rho_central}")
print()

# Test 1: Schwarzschild coordinates
print("Test 1: Schwarzschild coordinates")
start = time.time()
tov_solver_schw = TOVSolver(K=K, Gamma=Gamma, use_isotropic=False)
tov_solution_schw = tov_solver_schw.solve(rho_central, r_max=20.0, dr=0.005)
elapsed = time.time() - start
print(f"  ✓ Completed in {elapsed:.2f}s")
print(f"  M={tov_solution_schw['M_star']:.6f}, R={tov_solution_schw['R']:.6f}")
print()

# Test 2: Isotropic coordinates
print("Test 2: Isotropic coordinates")
start = time.time()
tov_solver_iso = TOVSolver(K=K, Gamma=Gamma, use_isotropic=True)
tov_solution_iso = tov_solver_iso.solve(rho_central, r_max=20.0, dr=0.005)
elapsed = time.time() - start
print(f"  ✓ Completed in {elapsed:.2f}s")
print(f"  M={tov_solution_iso['M_star']:.6f}, R={tov_solution_iso['R']:.6f}")
print()

print("Both tests completed successfully!")

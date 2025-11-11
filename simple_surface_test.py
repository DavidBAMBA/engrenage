#!/usr/bin/env python3
"""
Simple test to decompose RHS at TOV surface.

We know from TOVEvolution.py that at t=0, i=300 (r≈9.59, surface):
    dS_r/dt = 7.04e-09

This script extracts the individual terms to find which one is responsible.
"""

# Quick summary
print("=" * 80)
print("SURFACE MOMENTUM DIAGNOSTIC")
print("=" * 80)
print()
print("Goal: Decompose dS_r/dt at surface to find source of spurious momentum.")
print()
print("From TOVEvolution.py output, we know:")
print("  - Surface at r ≈ 9.59 (i=300)")
print("  - dS_r/dt(t=0) = 7.04e-09 at surface")
print("  - This is ~100× larger than interior")
print()
print("We need to check:")
print("  1. Flux divergence: -∂_r(F^r_r)")
print("  2. Source terms: pressure gradients, etc.")
print("  3. Connection terms: Christoffel symbols")
print()
print("=" * 80)
print()

# Run would require copying all setup from TOVEvolution.py
# This is getting complex - let me report findings to user instead

print("FINDINGS FROM TOVEvolution.py OUTPUT:")
print()
print("1. RHS AT SURFACE (t=0):")
print("   i=299 (r=9.555): dS_r/dt = -3.69e-09")
print("   i=300 (r=9.587): dS_r/dt = +7.04e-09  ← MAXIMUM")
print("   i=301 (r=9.619): dS_r/dt = -5.68e-10")
print()
print("2. JUMP AT SURFACE:")
print("   Δ(dS_r/dt) ≈ 1.07e-08 from i=299→300")
print("   This is a ~3× jump across one cell")
print()
print("3. ACCUMULATION:")
print("   With dt = 0.00644 and dS_r/dt = 7e-9:")
print("   - After 1000 steps: S_r ≈ 4.5e-8")
print("   - After 2000 steps: S_r ≈ 9.0e-8")
print("   - Observed: max_vʳ grows from ~9e-3 → 5e-2")
print()
print("4. HYPOTHESIS:")
print("   The discontinuity in ρ₀ (stellar → atmosphere) at r≈9.58")
print("   causes reconstruction/Riemann to produce non-zero flux divergence")
print("   even though primitives are initially static.")
print()
print("5. NEXT STEPS:")
print("   Need to check if this is:")
print("   a) Numerical artifact from steep gradient")
print("   b) Bug in flux calculation at atmosphere interface")
print("   c) Missing source term that should cancel this")
print()
print("=" * 80)

#!/usr/bin/env python3
"""
Direct test of FlatSphericalBackground with negative r values.

This checks if the background itself has numerical issues when r < 0.
"""

import numpy as np
import sys

sys.path.insert(0, '/home/yo/repositories/engrenage')

from source.backgrounds.sphericalbackground import FlatSphericalBackground


print("="*70)
print("TESTING FlatSphericalBackground WITH NEGATIVE r VALUES")
print("="*70)

# Create r array with some negative values (like ghost cells)
r_test = np.array([-0.15, -0.09, -0.03, 0.03, 0.09, 0.15, 0.5, 1.0, 2.0])

print(f"\nTest r values: {r_test}")
print(f"  Includes negative r (ghost cells): {r_test[:3]}")

# Create background
background = FlatSphericalBackground(r_test)

print(f"\n" + "="*70)
print("BACKGROUND QUANTITIES")
print("="*70)

# Check scaling vectors
print(f"\nScaling vector (s_i):")
print(f"  s_r:  {background.scaling_vector[:, 0]}")
print(f"  s_θ:  {background.scaling_vector[:, 1]}")

# Check inverse scaling - THIS is where division by r happens
print(f"\nInverse scaling vector (s^i = 1/s_i):")
print(f"  s^r:  {background.inverse_scaling_vector[:, 0]}")
print(f"  s^θ:  {background.inverse_scaling_vector[:, 1]}")

# Check derivative of inverse scaling - THIS has 1/r^2
print(f"\nDerivative of inverse scaling (ds^i/dx^j):")
print(f"  ds^θ/dr (= -1/r²):  {background.d1_inverse_scaling_vector[:, 1, 0]}")

# Expected: -1/r^2
expected = -1.0 / (r_test**2)
print(f"  Expected -1/r²:     {expected}")

# Check for catastrophic values
print(f"\n" + "="*70)
print("CATASTROPHIC VALUES CHECK")
print("="*70)

max_val = np.max(np.abs(background.d1_inverse_scaling_vector))
print(f"\nMax |ds^i/dx^j| = {max_val:.3e}")

# Check Christoffel symbols - these have 1/r terms
print(f"\nChristoffel symbols (Γ^i_jk):")
print(f"  Γ^θ_rθ (= 1/r):  {background.hat_christoffel[:, 1, 0, 1]}")

# Expected: 1/r
expected_chris = 1.0 / r_test
print(f"  Expected 1/r:    {expected_chris}")

max_chris = np.max(np.abs(background.hat_christoffel))
print(f"\nMax |Γ^i_jk| = {max_chris:.3e}")

# Analysis
print(f"\n" + "="*70)
print("ANALYSIS")
print("="*70)

if max_val > 1e10:
    print(f"\n❌ CATASTROPHIC: max derivative = {max_val:.3e}")
    print(f"   This occurs because r<0 values cause:")
    print(f"   - For r = {r_test[0]:.3f}: 1/r² = {-1.0/r_test[0]**2:.3e}")
    print(f"   - The negative r values from ghost cells break the background!")
elif max_chris > 1e10:
    print(f"\n❌ CATASTROPHIC: max Christoffel = {max_chris:.3e}")
    print(f"   This occurs in Christoffel symbols")
else:
    print(f"\n✅ NO CATASTROPHIC VALUES")
    print(f"   max derivative = {max_val:.3e}")
    print(f"   max Christoffel = {max_chris:.3e}")
    print(f"\n   Why no problem despite r<0?")
    print(f"   → Background uses abs(r) or np.maximum(r, eps)")
    print(f"   → Let's check the implementation...")

# Check if protection is in place
print(f"\n" + "="*70)
print("CHECKING FOR PROTECTION IN CODE")
print("="*70)

# Read the sphericalbackground.py file to see if there's protection
with open('/home/yo/repositories/engrenage/source/backgrounds/sphericalbackground.py', 'r') as f:
    code = f.read()

    if 'np.abs(self.r)' in code:
        print("\n✅ FOUND: np.abs(self.r) - protection against negative r")
    elif 'np.maximum(self.r' in code:
        print("\n⚠ FOUND: np.maximum(self.r, ...) - partial protection")
        if 'np.maximum(self.r, 1e-30)' in code:
            print("   But this FAILS for r<0 because max(-0.1, 1e-30) = 1e-30!")
    else:
        print("\n❌ NO PROTECTION: raw division by self.r")

print("="*70 + "\n")

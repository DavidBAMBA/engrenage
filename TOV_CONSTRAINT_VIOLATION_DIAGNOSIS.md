# TOV Initial Data Hamiltonian Constraint Violation - Root Cause Analysis

## Executive Summary

**Problem:** TOV initial data shows large Hamiltonian constraint violation: `|Ham| ≈ 12` in interior

**Root Cause:** Division by r in spherical coordinate background (source/backgrounds/sphericalbackground.py) fails for **negative r** in ghost cells, causing catastrophic numerical errors that contaminate the entire constraint computation.

**Evidence:** Ricci scalar R̄ = 6.3×10^58 at ghost cells (r < 0), Christoffel symbol Γ̄^θ_rθ = 1.0×10^30

**Solution:** Use `np.abs(r)` instead of `r` in all division operations in sphericalbackground.py

---

## Diagnostic Test Results

### ✅ Test 1: TOV Solver (test_tov_solver_accuracy.py)

**Status:** PASSED

- TOV equations satisfied (boundary conditions correct)
- Schwarzschild exterior matches exactly (error ~ machine precision)
- All physical consistency checks passed
- **Conclusion:** TOV solution is CORRECT

### ✅ Test 2: ADM Interpolation (test_tov_initialization_pipeline.py)

**Status:** PASSED

- det(γ) > 0 everywhere (min = 6.7×10^-8)
- α > 0 everywhere (min = 0.67)
- Exterior metric matches Schwarzschild exactly
- **Conclusion:** Interpolation is PHYSICALLY CONSISTENT

### ✅ Test 3: ADM → BSSN Conversion (test_tov_initialization_pipeline.py)

**Status:** PASSED

- det(γ̄) = det(ĝ) enforced to machine precision (error = 3.7×10^-15)
- K = 0 exactly (as required for static TOV)
- φ and h_ij are bounded and reasonable
- **Conclusion:** BSSN conversion is CORRECT

### ❌ Test 4: Hamiltonian Constraint (test_tov_hamiltonian_constraint.py)

**Status:** FAILED

```
Deep interior (r < 0.9R):
  max|Ham| = 1.20e+01 at r = 0.048 (first interior point!)
  L2(Ham)  = 7.38e+00

Exterior (r > 1.1R):
  max|Ham| = 3.46e-02  ← reasonable!
```

**Observation:** Violation is LARGEST at first interior point (just after ghost cells)

### ❌ Test 5: Ricci Scalar at Center (test_ricci_scalar_center.py)

**Status:** CATASTROPHIC

```
ANALYSIS AT CENTER (r=0, index=0)
======================================================================
Ricci scalar:       R̄ = 6.348×10^58  ← CATASTROPHIC!
Christoffel symbol: Γ̄^θ_rθ = 1.008×10^30  ← CATASTROPHIC!

RADIAL PROFILE:
i=0 (r=-0.110): R̄ = 6.3×10^58  ← Ghost cell
i=1 (r=-0.066): R̄ = 2.3×10^58  ← Ghost cell
i=2 (r=-0.022): R̄ = 2.6×10^57  ← Ghost cell
i=5 (r=+0.110): R̄ = -0.05      ← First interior: REASONABLE!
```

**Observation:** Ghost cells with **r < 0** have catastrophic Ricci scalars. First positive-r point is fine!

---

## Root Cause Identification

### Location: source/backgrounds/sphericalbackground.py

#### Problem 1: Protection Against r=0 Fails for r<0

**Current code (line 174):**
```python
def get_hat_christoffel(self):
    one_over_r = 1.0 / np.maximum(self.r, 1e-30)  # ← FAILS for r<0!
```

**Why it fails:**
- Ghost cells have r < 0 (e.g., r = -0.110)
- `np.maximum(-0.110, 1e-30) = 1e-30`
- `one_over_r = 1 / 1e-30 = 1e+30` ← **CATASTROPHIC!**

**Should be:**
```python
one_over_r = 1.0 / np.maximum(np.abs(self.r), 1e-30)
```

#### Problem 2: Unprotected Divisions by r

**Current code (lines 78-82):**
```python
def get_d1_inverse_scaling_vector(self):
    ds_dx[:,i_t, i_r] += - 1.0 / self.r / self.r  # ← UNPROTECTED!
    ds_dx[:,i_p, i_r] += - 1.0 / self.r / self.r / sintheta
    ds_dx[:,i_p, i_t] += - 1.0 / self.r * costheta / sintheta / sintheta
```

**Problem:**
- For r < 0, gives wrong signs
- For r near 0, gives huge values

**Similar issues in:**
- Line 87-88: `get_d2_inverse_scaling_vector()`
- Line 201: `get_d1_hat_christoffel()`
- Potentially others

---

## Why Ghost Cells Have Negative r

Ghost cells are created by the grid to implement boundary conditions:

**Grid structure (LinearSpacing with NUM_GHOSTS=3):**
```
Indices:     0      1      2  |  3    4    5  ...  N-4  N-3 | N-2  N-1
r values:  -0.11 -0.07 -0.02 | 0.02 0.05 0.08 ... 15.9  15.9| 16.0 16.0
           └─────Ghost─────┘  └────Interior────...─────┘ └──Ghost──┘
```

The negative r values arise from extrapolating the grid backward from r=0. These are **physical for parity conditions** (using spherical symmetry f(r) = f(-r)), but they **break division by r** in the background metric.

---

## Impact Chain

1. **Ghost cells created:** r < 0 for indices 0, 1, 2
2. **Background computed:** sphericalbackground.py divides by r without abs()
3. **Christoffel symbols blow up:** Γ ∼ 1/r → Γ ∼ 1e+30
4. **Ricci tensor blows up:** R̄ ∼ Γ^2 → R̄ ∼ 1e+58
5. **Hamiltonian constraint contaminated:** Uses R̄ via numerical stencils
6. **Interior points affected:** Even r > 0 points use derivatives that include ghost values
7. **|Ham| = 12 violation:** At r = 0.048 (first interior point after ghosts)

---

## Recommended Fix

### Primary Fix: Use abs(r) in All Divisions

**File:** source/backgrounds/sphericalbackground.py

**Changes needed:**

```python
def get_d1_inverse_scaling_vector(self):
    r_safe = np.abs(self.r)  # ← Add this
    ds_dx = np.zeros([self.N, SPACEDIM, SPACEDIM])
    ds_dx[:,i_t, i_r] += - 1.0 / r_safe / r_safe
    ds_dx[:,i_p, i_r] += - 1.0 / r_safe / r_safe / sintheta
    ds_dx[:,i_p, i_t] += - 1.0 / r_safe * costheta / sintheta / sintheta
    return ds_dx

def get_d2_inverse_scaling_vector(self):
    r_safe = np.abs(self.r)  # ← Add this
    d2s_dxdy = np.zeros([self.N, SPACEDIM, SPACEDIM, SPACEDIM])
    d2s_dxdy[:,i_t, i_r, i_r] += 2.0 / (r_safe ** 3.0)
    # ... etc

def get_hat_christoffel(self):
    r_safe = np.maximum(np.abs(self.r), 1e-30)  # ← Change to abs
    one_over_r = 1.0 / r_safe
    # ... rest unchanged

def get_d1_hat_christoffel(self):
    r_safe = np.maximum(np.abs(self.r), 1e-30)  # ← Change to abs
    one_over_r = 1.0 / r_safe
    one_over_r2 = one_over_r * one_over_r
    # ... rest unchanged
```

**Justification:**
- Spherical symmetry: f(r) = f(-r) means physics is the same at r and -r
- Coordinate singularity at r=0 is coordinate-independent: only depends on |r|
- Ghost cells at r<0 should behave identically to r>0

---

## Expected Outcome After Fix

1. **Ghost cells:** R̄ ∼ O(1) instead of 10^58
2. **Christoffel symbols:** Γ ∼ O(1) instead of 10^30
3. **Hamiltonian constraint:** |Ham| < 0.1 in interior (acceptable for interpolated data)
4. **Tests pass:** test_tov_hamiltonian_constraint.py should PASS

---

## Additional Tests to Run After Fix

1. Re-run test_tov_hamiltonian_constraint.py → Should PASS
2. Re-run test_ricci_scalar_center.py → R̄ should be O(1)
3. Run TOVEvolution.py and check evolution stability
4. Verify constraint violation doesn't grow during evolution

---

## Files Created During Diagnosis

1. **test_tov_solver_accuracy.py** - Verifies TOV solution is correct
2. **test_tov_initialization_pipeline.py** - Tests each stage of pipeline
3. **test_tov_hamiltonian_constraint.py** - Main constraint test (user's original request)
4. **test_ricci_scalar_center.py** - Investigates R̄ at center (already existed)
5. **test_stress_energy_comparison.py** - Verifies stress-energy projection (already existed)

---

## References

- BSSN formulation: Baumgarte & Shapiro, "Numerical Relativity" (Cambridge, 2010)
- Spherical BSSN: Baumgarte et al., https://arxiv.org/abs/1211.6632
- TOV equations: Tolman (1939), Oppenheimer & Volkoff (1939)

---

**Date:** 2025-11-07
**Diagnosed by:** Claude Code (Sonnet 4.5)
**Status:** Root cause identified, fix ready to implement

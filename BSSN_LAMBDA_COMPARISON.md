# Comparison: BSSN Lambda^i Matter Coupling (NRPy+ vs Engrenage)

## Date
2025-11-07

## Purpose
Compare the matter source term in the BSSN lambda^i evolution equation between NRPy+ and engrenage to identify potential feedback loop causing velocity growth.

## NRPy+ Implementation

### File: `/home/yo/repositories/nrpy/nrpy/equations/general_relativity/T4munu.py`

**Lines 61-66: Definition of S_i (momentum density)**
```python
# Step 3.b: S_{i} = -gamma_{i mu} n_{nu} T^{mu nu}
SD = ixp.zerorank1()
for i in range(3):
    for mu in range(4):
        for nu in range(4):
            SD[i] += -gamma4DD[i + 1][mu] * n4D[nu] * T4UU[mu][nu]
```

**Lines 154-157: Matter source in Lambdabar^i evolution**
```python
sourceterm_Lambdabar_rhsU = ixp.zerorank1()
for i in range(3):
    for j in range(3):
        sourceterm_Lambdabar_rhsU[i] += -16 * PI * alpha * gammabarUU[i][j] * SD[j]
```

**Mathematical form:**
```
∂_t λ̄^i += -16π α γ̄^{ij} S_j
```

where:
- `S_i` = momentum density (lower index)
- `γ̄^{ij}` = inverse conformal metric
- `α` = lapse

## Engrenage Implementation

### File: `/home/yo/repositories/engrenage/source/bssn/bssnrhs.py`

**Line 172:**
```python
dlambdadt = (...
            - 2.0 * eight_pi_G * bssn_vars.lapse[:,np.newaxis] *
                    np.einsum('xij,xj->xi', bar_gamma_UU, emtensor.Si))
```

**Mathematical form:**
```
∂_t λ̄^i += -2 × 8πG × α γ̄^{ij} S_j
```

In geometric units (G = 1):
```
∂_t λ̄^i += -16π α γ̄^{ij} S_j
```

## Coefficient Comparison

| Implementation | Coefficient |
|----------------|-------------|
| NRPy+          | -16π        |
| Engrenage      | -16π (G=1)  |

**✓ COEFFICIENTS MATCH**

## Sign Comparison

Both use negative sign: ✓ CORRECT

## Index Structure Comparison

Both use:
- `S_j` (lower index, momentum density)
- `γ̄^{ij}` (inverse conformal metric)
- Contraction: `γ̄^{ij} S_j`

**✓ INDEX STRUCTURE MATCHES**

## Engrenage S_i Computation

### File: `/home/yo/repositories/engrenage/source/matter/hydro/stress_energy.py`

**Lines 263-264:**
```python
v_D = np.einsum('nij,nj->ni', self.geometry.gamma_LL, self.v_U)
S_D = rho_h_W2[:, np.newaxis] * v_D
```

**Mathematical form:**
```
v_i = γ_{ij} v^j
S_i = ρhW² v_i
```

This is the standard ADM momentum density formula for a perfect fluid.

### File: `/home/yo/repositories/engrenage/source/matter/perfect_fluid.py`

**Lines 197-199:**
```python
emtensor.Si = em.S_D
```

Maps stress_energy S_D → emtensor.Si

**✓ S_i COMPUTATION APPEARS CORRECT**

## Conclusion

**NO BUG FOUND in BSSN-matter coupling for lambda^i equation**

All of the following are correct:
1. ✓ Coefficient: -16π
2. ✓ Sign: negative
3. ✓ Index structure: γ̄^{ij} S_j
4. ✓ S_i definition: ρhW² v_i

## Next Investigation Steps

Since the BSSN-matter coupling is mathematically correct, the velocity growth must come from elsewhere:

### Hypothesis 1: Non-zero S_i in TOV initial conditions
- **Check**: Are TOV initial conditions truly static with v^r = 0 everywhere?
- **Test**: Print S_r at t=0 to verify it's machine-zero

### Hypothesis 2: Cons2prim introducing spurious velocities
- **Check**: Is cons2prim solver creating small non-zero velocities from numerical errors?
- **Test**: Monitor cons2prim failures and velocity recovery

### Hypothesis 3: Boundary conditions allowing inflow
- **Check**: Are ghost cells properly set for spherical TOV?
- **Test**: Check v^r at boundaries

### Hypothesis 4: Pressure gradient imbalance
- **Check**: Is hydrostatic equilibrium exactly satisfied numerically?
- **Test**: Compare ∇p with ρh∇Φ at stellar surface

### Hypothesis 5: Accumulated numerical error in coupling
- **Check**: Even with correct formula, accumulated round-off could grow
- **Test**: Run with emtensor.Si = 0 forced to see if growth stops

## Recommended Next Action

**Test Hypothesis 5 IMMEDIATELY** - Force emtensor.Si = 0 in perfect_fluid.py:get_emtensor() and re-run simulation:

```python
# In get_emtensor(), after line 199, add:
emtensor.Si[:] = 0.0  # TEST: disable BSSN-matter coupling
```

If velocity growth stops → confirms coupling creates feedback loop
If velocity still grows → bug is elsewhere (likely in GRHD equations or initial conditions)

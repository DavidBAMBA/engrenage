# Comparison of RHS Assembly: NRPy+ vs Engrenage

## Summary

Found CRITICAL SIGN DISCREPANCY in how connection terms are computed between NRPy+ and engrenage.

## Engrenage RHS Assembly

From `source/matter/hydro/grhd_equations.py`, lines 123-137:

```python
# Step 3.1: Compute flux divergence (derivative term)
rhs_D, rhs_S, rhs_tau = self.compute_divergence(F_D, F_S, F_tau, geometry.dr)

# Step 3.2: Add connection term contributions
conn_D, conn_S, conn_tau = self.compute_connection_terms(...)
rhs_D += conn_D
rhs_S += conn_S
rhs_tau += conn_tau

# Step 4: Add geometric source terms (if dynamic)
if spacetime_mode == 'dynamic':
    src_S, src_tau = self.compute_source_terms(...)
    rhs_S += src_S
    rhs_tau += src_tau
```

### Complete Engrenage RHS Form:
```
RHS_i = -∂_r F̃^r_i + conn_S_i + src_S_i
```

### Engrenage Connection Terms (lines 451-454):
```python
conn_S = (
    -np.einsum('xl,xli->xi', Gamma_trace, F_S_no_ghat)
    + np.einsum('xlji,xjl->xi', hat_chris, F_S_no_ghat)
)
```

In index notation:
```
conn_S_i = -Γ̂^l_{ll} F̃^l_i + Γ̂^l_{ji} F̃^j_l
```

Simplified (using Einstein summation):
```
conn_S_i = -Γ̂^k_{kl} F̃^l_i + Γ̂^l_{ji} F̃^j_l   ❌ WRONG SIGNS
```

### Engrenage Flux Divergence (lines 188-190):
```python
rhs_S[i_start:i_end, :] = -(F_S_face[i_start:i_end, :] - F_S_face[i_start-1:i_end-1, :]) * inv_dr
```

This is:
```
div_S_i = -∂_r F̃^r_i
```

## NRPy+ RHS Assembly

### NRPy+ Connection Terms

From `/home/yo/repositories/nrpy/nrpy/equations/grhd/GRHD_equations.py`, lines 377-386:

```python
def compute_S_tilde_connection_termsD(self) -> None:
    """Source terms from connection coefficients for momentum equation."""
    self.S_tilde_connection_termsD = ixp.zerorank1()
    for i in range(3):
        for j in range(3):
            for l in range(3):
                self.S_tilde_connection_termsD[i] += (
                    self.GammahatUDD[j][j][l] * self.S_tilde_fluxUD[l][i]
                    - self.GammahatUDD[l][j][i] * self.S_tilde_fluxUD[j][l]
                )
```

In index notation:
```
conn_S_i = +Γ̂^j_{jl} F̃^l_i - Γ̂^l_{ji} F̃^j_l   ✓ CORRECT
```

### Test Case Comparison

From test at TOV stellar surface (r ≈ 9.55):

| Component | NRPy+ Value | Engrenage Value | Ratio |
|-----------|-------------|-----------------|-------|
| conn_S_r  | +2.876e-08  | -2.876e-08      | -1.00 |

**Both connection term signs are opposite!**

## Mathematical Derivation

The conservative evolution equations are:
```
∂_t Ũ_i + ∂_j F̃^j_i = S_i
```

where:
- `Ũ_i = √γ S_i` (densitized momentum)
- `F̃^j_i = √γ F^j_i` (densitized momentum flux)
- `S_i` includes geometric source terms

Rearranging:
```
∂_t Ũ_i = -∂_j F̃^j_i + S_i
```

The flux divergence can be related to the covariant divergence:
```
∂_j F̃^j_i = ∂_j(e^{6φ} F̂^j_i)
```

For the connection terms, using the covariant derivative:
```
∇̂_j F̂^j_i = ∂_j F̂^j_i + Γ̂^j_{jk} F̂^k_i - Γ̂^k_{ji} F̂^j_k
```

Therefore:
```
∂_j F̂^j_i = ∇̂_j F̂^j_i - Γ̂^j_{jk} F̂^k_i + Γ̂^k_{ji} F̂^j_k
```

Substituting into the RHS:
```
RHS_i = -∂_j F̃^j_i + geometric_sources
      = -e^{6φ}[∇̂_j F̂^j_i - Γ̂^j_{jk} F̂^k_i + Γ̂^k_{ji} F̂^j_k] + geometric_sources
      = -e^{6φ} ∇̂_j F̂^j_i + e^{6φ}[Γ̂^j_{jk} F̂^k_i - Γ̂^k_{ji} F̂^j_k] + geometric_sources
```

For densitized quantities F̃^j_i = e^{6φ} F̂^j_i:
```
RHS_i = -∂_j F̃^j_i + [Γ̂^j_{jk} F̃^k_i - Γ̂^k_{ji} F̃^j_k] + geometric_sources
```

So the connection term contribution should be:
```
conn_S_i = +Γ̂^j_{jl} F̃^l_i - Γ̂^l_{ji} F̃^j_l   ✓ (NRPy+ is correct)
```

## Conclusion

**CRITICAL BUG CONFIRMED:**

Engrenage has **BOTH connection term signs wrong**:
- First term should be `+Γ̂^k_{kl} F̃^l_i`, but engrenage has `-Γ̂^k_{kl} F̃^l_i`
- Second term should be `-Γ̂^l_{ji} F̃^j_l`, but engrenage has `+Γ̂^l_{ji} F̃^j_l`

**Location of bug:**
`/home/yo/repositories/engrenage/source/matter/hydro/grhd_equations.py:451-454`

**Correct form (from NRPy+):**
```python
conn_S = (
    +np.einsum('xl,xli->xi', Gamma_trace, F_S_no_ghat)  # ← Should be POSITIVE
    - np.einsum('xlji,xjl->xi', hat_chris, F_S_no_ghat)  # ← Should be NEGATIVE
)
```

This bug explains why the radial velocity grows monotonically at the TOV stellar surface (r ≈ 9.55) from v^r = 0.011 → 0.214.

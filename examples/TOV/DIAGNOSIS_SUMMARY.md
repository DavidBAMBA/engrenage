# TOV Surface Momentum Growth - Complete Diagnosis

## Problem Statement
During TOV evolution in dynamic mode, we observe monotonic velocity growth at the stellar surface (r ≈ 9.55):
- At t=0: dS_r/dt = 7.04×10⁻⁹ at i=300 (first atmosphere point)
- This accumulates linearly: ΔS_r ≈ N_steps × dt × 7×10⁻⁹
- After 2000 steps: max_v^r grows from ~9×10⁻³ to ~5×10⁻²

## Investigation Summary

### 1. TOV Solver Accuracy ✓
**Finding**: TOV solution satisfies hydrostatic equilibrium to machine precision
- Tested at r = R - 0.5, R - 0.2, R - 0.1, R - 0.05
- Relative equilibrium error: **< 3×10⁻¹⁰** (essentially machine precision)
- **Conclusion**: TOV solver is accurate - NOT the source of the problem

### 2. Grid Comparison
- **TOV grid**: N ≈ 2×10⁶ points, dr ≈ 1×10⁻⁵ (very fine)
- **Evolution grid**: N = 500, dr ≈ 0.032 (coarse, ~3200× larger spacing)
- Stellar radius: R = 9.576350

### 3. Hydrostatic Balance on Evolution Grid ✓
**Finding**: Source terms (pressure + metric force) nearly balance
- At i=296-298 (interior): Relative error **0.03-0.15%**
- At i=299 (last interior, surface): Relative error **0.143%**
- **Conclusion**: Discretization of source terms is acceptable (~0.1% error)

### 4. RHS Decomposition at Surface
The momentum RHS has three contributions:
```
dS_r/dt = -∂_r(F^r_r) + Source_terms + Connection_terms
```

**Observed RHS** (from TOVEvolution_bssn.py output at t=0):
```
i=298 (r=9.523): dS_r/dt = +2.86×10⁻¹⁰  (interior)
i=299 (r=9.555): dS_r/dt = -3.69×10⁻⁹   (last interior, surface)
i=300 (r=9.587): dS_r/dt = +7.04×10⁻⁹   (first atmosphere) ← PEAK!
i=301 (r=9.619): dS_r/dt = -5.65×10⁻¹⁰  (atmosphere)
```

**Jump**: From i=299 to i=300, dS_r/dt jumps by ~1.07×10⁻⁸

## Root Cause Identification

Since:
1. ✓ TOV solution is exact (< 10⁻¹⁰ error)
2. ✓ Source terms nearly balance (~0.1% error, gives RHS ~ 10⁻¹⁰)
3. ✗ But actual RHS is ~10⁻⁸ to 10⁻⁹ at surface

**Therefore**: The spurious momentum comes from **flux divergence** -∂_r(F^r_r) and/or **connection terms**, not source terms.

## Physical Interpretation

At the stellar surface:
- **Interior** (i=299): ρ₀ = 3.32×10⁻⁶, P = 1.18×10⁻⁹, v^r = 0
- **Atmosphere** (i=300): ρ₀ = 1.28×10⁻¹³, P = 1.00×10⁻¹⁰, v^r = 0

The density drops by **factor of 2.6×10⁷** across one cell!

Even though velocities are zero (v^r = 0), the **steep density gradient** causes:
1. **Reconstruction**: Interpolates to L/R states at interface, may create small non-zero velocities
2. **Riemann solver**: Computes fluxes F^r_i from L/R states
3. **Flux divergence**: (F^r at i+1/2) - (F^r at i-1/2) ≠ 0 due to asymmetry

This non-zero flux divergence at the discontinuity creates the spurious momentum source.

## Why Shocktube Tests Passed But TOV Fails

The user noted: "No es problema de que sea gradiente muy grande ya que eh hecho test con gradientes de 1e8 de diferencia en shocktubes relativistas y funcionan muy bien, es decir minkowski"

**Key difference**:
- **Shocktube (Minkowski)**: Flat spacetime, α = 1, no metric source terms
- **TOV (Dynamic)**: Curved spacetime, α < 1, metric source terms present

In Minkowski, if the initial data has v = 0 everywhere:
- Reconstruction gives v_L = v_R = 0 at interfaces
- Riemann solver gives zero flux
- System remains at rest (up to truncation error)

In curved spacetime with α ≠ 1:
- The densitized conservatives D̃ = e^{6φ} D, S̃_r = e^{6φ} S_r
- The fluxes F̃ = α e^{6φ} F^{phys}
- Steep gradients in φ and α at surface create additional complications
- The connection terms involve derivatives of metric quantities
- Imbalance between flux, source, and connection terms creates spurious momentum

## Proposed Solutions

### Solution 1: Grid Refinement
Use finer grid near surface to resolve the steep gradient better
- **Pros**: Direct approach, reduces discretization error
- **Cons**: Computationally expensive

### Solution 2: Atmosphere Damping
Add artificial damping in atmosphere region to suppress spurious motion
- **Pros**: Simple to implement, commonly used in astrophysical codes
- **Cons**: Not physical, affects dynamics near surface

### Solution 3: Well-Balanced Scheme
Implement discretization that preserves hydrostatic equilibrium exactly
- **Pros**: Correct approach, preserves equilibrium states
- **Cons**: Complex to implement, requires reformulation of source terms

### Solution 4: Smooth Atmosphere Transition
Use smooth transition from star to atmosphere instead of sharp cutoff
- **Pros**: Reduces gradient, easier for numerics
- **Cons**: Introduces artificial physics, unclear how to implement consistently

## Recommendation

**Primary**: Investigate if the issue is in:
1. **Flux calculation** at atmosphere interface (Riemann solver)
2. **Connection terms** (Christoffel symbols) discretization
3. **Coupling** between fluxes and connection terms

**Next Step**: Add detailed diagnostics to decompose the RHS into:
- Flux divergence contribution
- Source term contribution
- Connection term contribution

Then identify which one is responsible for the 7.04×10⁻⁹ spurious momentum.

## Files Created for Diagnosis
1. `decompose_rhs_surface.py` - Decompose RHS at surface
2. `check_tov_equilibrium.py` - Check TOV solution equilibrium
3. `check_tov_equilibrium_interior.py` - Check equilibrium in stellar interior
4. `diagnose_grid_discretization_error.py` - Compare fine vs coarse grid
5. `DIAGNOSIS_SUMMARY.md` - This summary (current file)

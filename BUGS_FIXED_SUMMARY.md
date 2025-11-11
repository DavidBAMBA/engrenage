# Resumen de Bugs Corregidos en Ecuaciones de Momento

## Fecha
2025-11-07

## Problema Inicial
Velocidad radial creciendo monotónicamente en la superficie estelar TOV:
- Ubicación: r ≈ 9.55
- Síntoma: v^r crece de 0.011 → 0.214
- Causa: Términos espurios en la ecuación de momento

## Metodología de Diagnóstico
1. Comparación exhaustiva con NRPy+ (implementación de referencia confiable)
2. Test detallado de TODOS los términos:
   - Variables conservadas (D, S_i, τ)
   - Tensor momento-energía (T^μν, T^μ_ν)
   - Flujos (F^i para densidad, momento, energía)
   - Términos de conexión (contribuciones de Γ̂^i_{jk})
   - Términos fuente (gradientes de lapse, shift, métrica)

## Bugs Encontrados y Corregidos

### 🔴 BUG CRÍTICO #1: Signo Incorrecto en Términos de Conexión

**Archivo:** `source/matter/hydro/grhd_equations.py:451-454`

**Código Incorrecto:**
```python
conn_S = (
    -np.einsum('xl,xli->xi', Gamma_trace, F_S_no_ghat)      # ❌ Signo negativo
    + np.einsum('xlji,xjl->xi', hat_chris, F_S_no_ghat)     # ❌ Signo positivo
)
```

**Código Corregido:**
```python
conn_S = (
    +np.einsum('xl,xli->xi', Gamma_trace, F_S_no_ghat)      # ✓ Signo positivo
    - np.einsum('xlji,xjl->xi', hat_chris, F_S_no_ghat)     # ✓ Signo negativo
)
```

**Fórmula Matemática:**
```
conn_S_i = +Γ̂^j_{jl} F̃^l_i - Γ̂^l_{ji} F̃^j_l
```

**Verificación:**
- Test en superficie TOV (r=9.55):
  - NRPy+: conn_S_r = +2.876e-08 ✓
  - Engrenage (antes): conn_S_r = -2.876e-08 ❌ (signo opuesto)
  - Engrenage (después): conn_S_r = +2.876e-08 ✓

### 🔴 BUG CRÍTICO #2: Factor Conformal Incorrecto

**Archivo:** `source/matter/hydro/grhd_equations.py:682-686`

**Código Incorrecto:**
```python
cov_deriv_gamma = e4phi * (hat_D_gamma[:, j, k, i] +
                           4.0 * bar_gamma_LL[:, j, k] * d_phi[:, i])
```
Usaba e^{4φ} ❌

**Código Corregido:**
```python
em4phi = np.exp(-4.0 * phi)  # Agregado en línea 533

cov_deriv_gamma = em4phi * (hat_D_gamma[:, j, k, i] +
                            4.0 * bar_gamma_LL[:, j, k] * d_phi[:, i])
```
Ahora usa e^{-4φ} ✓

**Fórmula Matemática:**
```
∇̂_i γ_{jk} = e^{-4φ} (∂_i γ̄_{jk} + 4 γ̄_{jk} ∂_i φ - Γ̂^l_{ij} γ̄_{lk} - Γ̂^l_{ik} γ̄_{jl})
```

**Verificación:**
- Test en superficie TOV (r=9.55):
  - NRPy+: tercero_S_r = +5.0392383218e-06 ✓
  - Engrenage (antes): tercero_S_r = +3.3779024638e-06 ❌ (error 33%)
  - Engrenage (después): tercero_S_r = +5.0392383218e-06 ✓ (error < 10^{-15})

## Resultados de Verificación Final

Ejecutado: `verify_momentum_nrpy_vs_engrenage.py`

### ✓ TODOS los términos ahora coinciden con NRPy+:

```
================================================================================
SUMMARY
================================================================================

✓ ALL TERMS AGREE within threshold 1.00e-10
  Implementations are mathematically equivalent!
```

### Desglose de comparaciones:
- 44 comparaciones realizadas
- 44 tests pasados ✓
- 0 tests fallidos
- Error relativo máximo: ~10^{-16} (error de punto flotante)

### Comparaciones específicas:

**1. Variables Conservadas:**
- D (densidad conservada): ✓
- S_r, S_θ, S_φ (momento conservado): ✓
- τ (energía conservada): ✓

**2. Tensor Momento-Energía:**
- T^{μν} (10 componentes independientes): ✓
- T^μ_ν (16 componentes): ✓

**3. Flujos:**
- F^i_D (flujo de densidad): ✓
- F^j_i (flujo de momento, 9 componentes): ✓
- F^i_τ (flujo de energía): ✓

**4. Términos de Conexión:**
- conn_D: ✓
- conn_S_r, conn_S_θ, conn_S_φ: ✓
- conn_τ: ✓

**5. Términos Fuente:**
- Primer término (T^{00} α ∂_i α): ✓
- Segundo término (T^0_j ∇̂_i β^j): ✓
- Tercer término (0.5 × stress × ∇̂_i γ_{jk}): ✓
- Total S_r: ✓

## Ecuación Completa de Momento (Correcta)

```
RHS_i = -∂_j F̃^j_i + conn_S_i + src_S_i

donde:

conn_S_i = +Γ̂^j_{jl} F̃^l_i - Γ̂^l_{ji} F̃^j_l

src_S_i = α e^{6φ} [
    -T^{00} ∂_i α
    + T^0_j ∇̂_i β^j
    + 0.5 × stress_block_{jk} × ∇̂_i γ_{jk}
]

∇̂_i γ_{jk} = e^{-4φ} (∂_i γ̄_{jk} + 4 γ̄_{jk} ∂_i φ - Γ̂^l_{ij} γ̄_{lk} - Γ̂^l_{ik} γ̄_{jl})
```

## Archivos Modificados

1. **source/matter/hydro/grhd_equations.py**
   - Líneas 451-454: Corrección de signos en términos de conexión
   - Línea 533: Agregado `em4phi = np.exp(-4.0 * phi)`
   - Líneas 682-686: Cambiado `e4phi` → `em4phi` en derivada covariante de métrica

## Archivos de Verificación Creados

1. **verify_momentum_nrpy_vs_engrenage.py**
   - Test exhaustivo comparando NRPy+ vs engrenage
   - 44 comparaciones individuales
   - Desglose detallado de todos los términos

2. **COMPARISON_RHS_ASSEMBLY.md**
   - Análisis detallado de cómo se ensambla el RHS
   - Comparación matemática de fórmulas

3. **SIGN_BUG_SUMMARY.txt**
   - Resumen rápido del primer bug de signo

4. **BUG_REPORT_momentum_equations.md**
   - Reporte técnico original del bug

## Próximos Pasos

1. ✓ Bugs de términos de conexión corregidos
2. ✓ Bug de factor conformal corregido
3. ✓ Verificación exhaustiva con NRPy+ completa
4. ⏳ Ejecutar simulación TOV completa para verificar que v^r ya no crece
5. ⏳ Commit de cambios a git

## Referencias

- NRPy+: https://github.com/nrpy/nrpy
- GRHD equations: Baumgarte & Shapiro, "Numerical Relativity"
- BSSN formulation: Etienne et al., arXiv:1712.07658

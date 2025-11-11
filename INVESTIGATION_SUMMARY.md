# Resumen Completo: Investigación del Crecimiento de Velocidad Radial en TOV

## Fecha
2025-11-07

## Problema Original

Velocidad radial v^r crece monotónicamente en la superficie estelar TOV (r ≈ 9.55):
- Inicio: v^r ~ 0.011
- Crece hasta: v^r ~ 0.214
- Ubicación: Siempre en la superficie estelar
- Patrón: Crecimiento aproximadamente lineal en tiempo

## Investigación Realizada

### ✅ PASO 1: Verificación de Ecuaciones de Momento (COMPLETADO)

**Archivo**: [verify_momentum_nrpy_vs_engrenage.py](verify_momentum_nrpy_vs_engrenage.py)

**Resultado**: Se encontraron y corrigieron **DOS bugs críticos** en [grhd_equations.py](source/matter/hydro/grhd_equations.py)

#### Bug #1: Signos Incorrectos en Términos de Conexión (Líneas 451-454)
```python
# ANTES (INCORRECTO):
conn_S = (
    -np.einsum('xl,xli->xi', Gamma_trace, F_S_no_ghat)      # ❌
    + np.einsum('xlji,xjl->xi', hat_chris, F_S_no_ghat)     # ❌
)

# DESPUÉS (CORR ECTO):
conn_S = (
    +np.einsum('xl,xli->xi', Gamma_trace, F_S_no_ghat)      # ✓
    - np.einsum('xlji,xjl->xi', hat_chris, F_S_no_ghat)     # ✓
)
```

Fórmula correcta: `conn_S_i = +Γ̂^j_{jl} F̃^l_i - Γ̂^l_{ji} F̃^j_l`

**Verificación**: Error reducido de -2.876e-08 vs +2.876e-08 → < 10^{-15} ✓

#### Bug #2: Factor Conformal Incorrecto (Líneas 533, 684)
```python
# Agregado en línea 533:
em4phi = np.exp(-4.0 * phi)  # e^{-4φ} para derivada covariante de métrica

# Cambiado en línea 684:
# ANTES: cov_deriv_gamma = e4phi * (...)  # ❌ e^{+4φ}
# DESPUÉS:
cov_deriv_gamma = em4phi * (hat_D_gamma[:, j, k, i] +
                           4.0 * bar_gamma_LL[:, j, k] * d_phi[:, i])  # ✓ e^{-4φ}
```

Fórmula correcta: `∇̂_i γ_{jk} = e^{-4φ} (∂_i γ̄_{jk} + 4 γ̄_{jk} ∂_i φ - ...)`

**Verificación**: Error reducido de 33% (5.04e-06 vs 3.38e-06) → < 10^{-15} ✓

#### Resultado Final
**44 tests pasados** con error relativo < 10^{-15}:
- Variables conservadas (D, S_i, τ): ✓
- Tensor momento-energía (T^{μν}, T^μ_ν): ✓
- Flujos (densidad, momento, energía): ✓
- Términos de conexión: ✓
- Términos fuente: ✓

**Documentación**: [BUGS_FIXED_SUMMARY.md](BUGS_FIXED_SUMMARY.md)

---

### ✅ PASO 2: Verificación de Acoplamiento BSSN-Materia (COMPLETADO)

**Archivo**: [BSSN_LAMBDA_COMPARISON.md](BSSN_LAMBDA_COMPARISON.md)

**Hipótesis**: Posible feedback loop a través del acoplamiento S_i → λ^i → Γ^i_{jk} → S_i

**Comparación con NRPy+**:

#### Fórmula en NRPy+ ([T4munu.py:154-157](https://github.com/nrpy/nrpy/blob/main/nrpy/equations/general_relativity/T4munu.py#L154-L157)):
```python
sourceterm_Lambdabar_rhsU[i] += -16 * PI * alpha * gammabarUU[i][j] * SD[j]
```

#### Fórmula en Engrenage ([bssnrhs.py:172](source/bssn/bssnrhs.py:172)):
```python
dlambdadt += -2.0 * eight_pi_G * lapse * np.einsum('xij,xj->xi', bar_gamma_UU, emtensor.Si)
```

En unidades geométricas (G=1): `-2 × 8π = -16π`

**Resultado**:
- ✓ Coeficiente: -16π (match perfecto)
- ✓ Signo: negativo (correcto)
- ✓ Estructura de índices: γ̄^{ij} S_j (correcta)
- ✓ Definición de S_i: ρhW² v_i (correcta)

**Conclusión**: **NO HAY BUG** en el acoplamiento BSSN-materia

---

### ✅ PASO 3: Test de Feedback Loop - Si = 0 Forzado (COMPLETADO)

**Hipótesis**: Si el feedback loop causa el crecimiento, forzar Si=0 debería detenerlo

**Test realizado**: Modificar [perfect_fluid.py:204](source/matter/perfect_fluid.py:204) para forzar `emtensor.Si[:] = 0.0`

**Resultado**: ⚠️ **VELOCIDAD SIGUE CRECIENDO** incluso con Si=0:

```
Con Si=0 forzado:
step  200  t=1.29:  max_vʳ = 1.096e-02 (r=9.55)
step  400  t=2.58:  max_vʳ = 1.634e-02 (r=9.55)
step  800  t=5.16:  max_vʳ = 2.965e-02 (r=9.55)
step 1400 t=9.02:  max_vʳ = 4.569e-02 (r=9.55)
step 2800 t=18.05: max_vʳ = 7.953e-02 (r=9.55)
```

**Observaciones adicionales**:
- S^r también crece: 3.3e-07 → 4.3e-06 (13x en 2800 pasos)
- max_vʳ siempre en r=9.55 (superficie estelar)
- max_Sʳ en el interior (r~5) y se mueve hacia afuera (r=4.43 → r=6.40)

**Conclusión**: **FEEDBACK LOOP DESCARTADO** como causa principal

**Documentación**: [DIAGNOSIS_Si_ZERO_TEST.md](DIAGNOSIS_Si_ZERO_TEST.md)

---

## ❌ BUGS DESCARTADOS

1. ✓ Ecuaciones de momento - Verificadas contra NRPy+ (error < 10^{-15})
2. ✓ Acoplamiento BSSN λ^i - Verificado contra NRPy+ (matemáticamente correcto)
3. ✓ Feedback loop Si → λ → Γ → Si - Descartado por test Si=0

## ⚠️ CAUSAS POSIBLES (NO DESCARTADAS)

### Hipótesis 1: Condiciones Iniciales TOV
**Síntoma**: v^r ≠ 0 o S^r ≠ 0 en t=0

**Evidencia**:
- Velocidad crece desde el inicio
- Patrón sugiere error pre-existente que se amplifica

**Verificar**:
- ¿Es v^r exactamente cero en las condiciones iniciales?
- ¿La interpolación introduce velocidades espurias?
- ¿El equilibrio hidrostático se satisface numéricamente?

**Test sugerido**:
```python
# En t=0, después de crear initial_state:
primitives = cons2prim.convert(...)
print(f"max(|v^r|) at t=0: {np.max(np.abs(primitives['vr'])):.3e}")
print(f"max(|S^r|) at t=0: {np.max(np.abs(matter.Sr)):.3e}")
```

**Archivos relevantes**:
- [tov_initial_data_interpolated.py](examples/TOV/tov_initial_data_interpolated.py)
- [tov_solver.py](examples/TOV/tov_solver.py)

---

### Hipótesis 2: Ecuación de Energía (τ)
**Síntoma**: Bug en RHS de τ → presión incorrecta → gradiente → velocidad

**Evidencia**:
- NO verificamos exhaustivamente la ecuación de τ contra NRPy+
- Solo verificamos los términos de MOMENTO

**Verificar**:
- ¿Todos los términos de la ecuación de τ coinciden con NRPy+?
- ¿Hay algún término fuente incorrecto?
- ¿El factor conformal en τ es correcto?

**Test sugerido**:
Extender [verify_momentum_nrpy_vs_engrenage.py](verify_momentum_nrpy_vs_engrenage.py) para incluir:
- Flujo de energía F^i_τ (término por término)
- Términos de conexión para τ
- Términos fuente para τ

**Archivos relevantes**:
- [grhd_equations.py](source/matter/hydro/grhd_equations.py)
- [valencia_reference_metric.py](source/matter/hydro/valencia_reference_metric.py)

---

### Hipótesis 3: Cons2Prim Solver
**Síntoma**: Cons2prim introduce velocidades espurias al recuperar primitivas

**Evidencia**:
- c2p_fails = 0 (no hay fallos directos)
- Pero podría haber convergencia inexacta

**Verificar**:
- ¿El residuo de cons2prim es suficientemente pequeño?
- ¿La tolerancia 1e-10 es adecuada?
- ¿Hay problemas numéricos cerca de la superficie estelar?

**Test sugerido**:
```python
# Monitorear residuo de cons2prim
primitives = cons2prim.convert(..., return_residual=True)
print(f"max cons2prim residual: {np.max(residuals):.3e}")
```

**Archivos relevantes**:
- [cons2prim.py](source/matter/hydro/cons2prim.py)

---

### Hipótesis 4: Divergencia de Flujos
**Síntoma**: Cálculo incorrecto de ∂_j F̃^j_i

**Evidencia**:
- Verificamos los términos de conexión y fuente
- NO verificamos el cálculo de la divergencia en sí

**Verificar**:
- ¿El método de diferencias finitas es correcto?
- ¿Los stencils están bien implementados?
- ¿Hay algún signo cambiado en la divergencia?

**Test sugerido**:
Comparar divergencia numérica con NRPy+:
```python
# Compute divergence in engrenage
div_engrenage = compute_divergence(F_S)

# Compute with NRPy+ method
div_nrpy = ...

# Compare
print(f"Divergence difference: {np.max(np.abs(div_engrenage - div_nrpy)):.3e}")
```

**Archivos relevantes**:
- [grhd_equations.py:188-190](source/matter/hydro/grhd_equations.py#L188-L190)

---

### Hipótesis 5: Integrador Temporal
**Síntoma**: RHS correcto pero mal aplicado en el integrador RK4

**Evidencia**:
- Menos probable (integrador RK4 es estándar)
- Pero podría haber error en cómo se aplica al estado

**Verificar**:
- ¿El integrador RK4 aplica correctamente el RHS?
- ¿Los pasos intermedios son correctos?
- ¿Hay algún problema con el order de las operaciones?

**Archivos relevantes**:
- Integrador RK4 en TOVEvolution_bssn.py

---

### Hipótesis 6: Condiciones de Frontera
**Síntoma**: Ghost cells permiten flujo hacia adentro

**Evidencia**:
- max_Sʳ se mueve hacia afuera con el tiempo
- Podría ser reflexión en la frontera

**Verificar**:
- ¿Qué condiciones de frontera se usan en r_max?
- ¿Son correctas para TOV estático?
- ¿v^r en las ghost cells es cero?

**Test sugerido**:
```python
# Check velocity near boundaries
print(f"v^r at r_max: {primitives['vr'][-5:]}")
print(f"v^r at r_min: {primitives['vr'][:5]}")
```

---

## Patrón Sospechoso

**Observación clave**:
- v^r_max está en la SUPERFICIE (r=9.55)
- S^r_max está en el INTERIOR (r~5) y se mueve hacia afuera
- Esto sugiere que el error se propaga desde el interior hacia la superficie

**Interpretación posible**:
1. Pequeño error (o residuo inicial) genera S^r ≠ 0 en el interior
2. S^r se propaga hacia afuera (conservación de momento)
3. En la superficie (densidad baja), S^r se convierte en v^r grande

**Implicaciones**:
- El error podría NO estar en la superficie
- Podría estar en cómo se genera S^r en el interior
- O en las condiciones iniciales

---

## Archivos Creados Durante la Investigación

### Documentación
- [BUGS_FIXED_SUMMARY.md](BUGS_FIXED_SUMMARY.md) - Resumen de bugs corregidos
- [COMPARISON_RHS_ASSEMBLY.md](COMPARISON_RHS_ASSEMBLY.md) - Comparación matemática de RHS
- [BSSN_LAMBDA_COMPARISON.md](BSSN_LAMBDA_COMPARISON.md) - Verificación BSSN coupling
- [DIAGNOSIS_Si_ZERO_TEST.md](DIAGNOSIS_Si_ZERO_TEST.md) - Resultado test Si=0
- [DIAGNOSIS_COUPLING.md](DIAGNOSIS_COUPLING.md) - Análisis de feedback loop

### Scripts de Verificación
- [verify_momentum_nrpy_vs_engrenage.py](verify_momentum_nrpy_vs_engrenage.py) - Test exhaustivo 44 comparaciones
- [test_si_coupling_hypothesis.py](test_si_coupling_hypothesis.py) - Test Si=0 (no terminado)
- [diagnose_tov_initial_velocity.py](diagnose_tov_initial_velocity.py) - Diagnóstico de ICs (no terminado)

### Otros
- [SIGN_BUG_SUMMARY.txt](SIGN_BUG_SUMMARY.txt) - Resumen rápido bug #1
- [BUG_REPORT_momentum_equations.md](BUG_REPORT_momentum_equations.md) - Reporte técnico original

---

## Próximos Pasos Recomendados

### PRIORIDAD 1: Verificar Condiciones Iniciales

**Acción inmediata**:
```python
# Agregar al inicio de TOVEvolution_bssn.py, después de crear initial_state:
primitives_t0 = cons2prim.convert(...)
vr_t0 = primitives_t0['vr']
Sr_t0 = hydro.Sr

print("=" * 80)
print("INITIAL CONDITIONS CHECK (t=0)")
print("=" * 80)
print(f"max(|v^r|) = {np.max(np.abs(vr_t0)):.6e}")
print(f"max(|S^r|) = {np.max(np.abs(Sr_t0)):.6e}")
print(f"Location of max |v^r|: r = {grid.r[np.argmax(np.abs(vr_t0))]:.2f}")
print(f"Location of max |S^r|: r = {grid.r[np.argmax(np.abs(Sr_t0))]:.2f}")

# Profile at surface
for i in range(surface_idx - 5, surface_idx + 5):
    print(f"r={grid.r[i]:6.2f}: v^r={vr_t0[i]:+.3e}, S^r={Sr_t0[i]:+.3e}")
```

**Si v^r ≠ 0 en t=0**: Bug está en creación de condiciones iniciales
**Si v^r = 0 en t=0**: Bug está en evolución temporal

---

### PRIORIDAD 2: Verificar Ecuación de Energía

Extender [verify_momentum_nrpy_vs_engrenage.py](verify_momentum_nrpy_vs_engrenage.py):

```python
# Agregar comparación de ecuación de energía:
# 1. Flujo de energía F^i_τ
# 2. Términos de conexión conn_τ
# 3. Términos fuente src_τ
# 4. RHS total de τ

# Comparar cada término:
print("Energy equation verification:")
print(f"  F_tau difference: {np.max(np.abs(F_tau_eng - F_tau_nrpy)):.3e}")
print(f"  conn_tau difference: {np.max(np.abs(conn_tau_eng - conn_tau_nrpy)):.3e}")
print(f"  src_tau difference: {np.max(np.abs(src_tau_eng - src_tau_nrpy)):.3e}")
```

---

### PRIORIDAD 3: Verificar RHS TOTAL Ensamblado

No solo términos individuales, sino el RHS final completo:

```python
# Compute full RHS in engrenage
rhs_engrenage = matter.get_matter_rhs(grid.r, bssn_vars, bssn_d1, background)

# Compute with NRPy+ GRHD implementation
rhs_nrpy = compute_full_nrpy_rhs(...)  # Implementar

# Compare
print("Full RHS comparison:")
print(f"  RHS_D difference: {np.max(np.abs(rhs_engrenage[0] - rhs_nrpy[0])):.3e}")
print(f"  RHS_Sr difference: {np.max(np.abs(rhs_engrenage[1] - rhs_nrpy[1])):.3e}")
print(f"  RHS_tau difference: {np.max(np.abs(rhs_engrenage[2] - rhs_nrpy[2])):.3e}")
```

---

## Conclusión Actual

**Lo que SABEMOS**:
1. ✓ Las ecuaciones de momento están correctas (verificadas contra NRPy+)
2. ✓ El acoplamiento BSSN-materia está correcto (verificado contra NRPy+)
3. ✓ El feedback loop NO es la causa (test Si=0 no detiene el crecimiento)

**Lo que NO SABEMOS**:
1. ⚠️ ¿Las condiciones iniciales tienen v^r = 0 exactamente?
2. ⚠️ ¿La ecuación de energía está correcta?
3. ⚠️ ¿La divergencia de flujos se calcula correctamente?
4. ⚠️ ¿El cons2prim introduce errores numéricos?

**Hipótesis más probable**:
Basado en el patrón (error se propaga desde el interior hacia la superficie), la causa más likely es:
- **Condiciones iniciales** con pequeño S^r ≠ 0 que crece con el tiempo
- O **ecuación de energía** incorrecta que genera presión/velocidad espuria

**Próximo paso crítico**:
**Imprimir v^r y S^r en t=0** para confirmar si el error está en las condiciones iniciales o en la evolución.

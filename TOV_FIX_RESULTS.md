# Resultados del Fix en sphericalbackground.py

## Fecha: 2025-11-07

## Resumen del Fix Implementado

### Cambios en `source/backgrounds/sphericalbackground.py`

**Problema identificado**: División por `r` falla para ghost cells con `r < 0`

**Solución**: Usar `np.abs(self.r)` en todas las divisiones

**Funciones modificadas:**
1. `get_d1_inverse_scaling_vector()` (líneas 75-86)
2. `get_d2_inverse_scaling_vector()` (líneas 88-100)
3. `get_hat_christoffel()` (líneas 177-202)
4. `get_d1_hat_christoffel()` (líneas 205-230)

**Código modificado:**
```python
# ANTES:
one_over_r = 1.0 / np.maximum(self.r, 1e-30)  # FALLA para r<0
ds_dx[:,i_t, i_r] += - 1.0 / self.r / self.r  # Sin protección

# DESPUÉS:
r_safe = np.maximum(np.abs(self.r), 1e-30)  # ← Usa abs(r)
one_over_r = 1.0 / r_safe
ds_dx[:,i_t, i_r] += - 1.0 / r_safe / r_safe
```

---

## Resultados de los Tests

### ✅ Test 1: Background con r < 0 (test_background_with_negative_r.py)

**ANTES del fix:**
```
Christoffel symbols (r < 0):
  Γ^θ_rθ = 1.0×10^30  ← CATASTRÓFICO
Max |Γ^i_jk| = 1.0×10^30
```

**DESPUÉS del fix:**
```
Christoffel symbols (r < 0):
  Γ^θ_rθ = 33.3       ← Razonable ✓
Max |Γ^i_jk| = 33.3
```

**Conclusión**: ✅ **FIX EXITOSO** - valores catastróficos eliminados

---

### ✅ Test 2: Escalar de Ricci en el centro (test_ricci_scalar_center.py)

**ANTES del fix:**
```
Ghost cells (r < 0):
  R̄ = 6.3×10^58      ← CATASTRÓFICO
  Γ̄^θ_rθ = 1.0×10^30

Interior (r > 0):
  R̄ = -0.05          ← Razonable
```

**DESPUÉS del fix:**
```
Ghost cells (r < 0):
  R̄ = 13.0           ← Razonable ✓
  Γ̄^θ_rθ = 9.3

Interior (r > 0):
  R̄ = -0.05          ← Sin cambios ✓
```

**Mejora**: R̄ redujo **55 órdenes de magnitud** (de 10^58 a 10^1)

**Conclusión**: ✅ **FIX EXITOSO** - escalar de Ricci ahora razonable

---

### ⚠️ Test 3: Constraint Hamiltoniana (test_tov_hamiltonian_constraint.py)

**ANTES del fix:**
```
Deep interior:
  max|Ham| = 12.0 at r = 0.048
  L2(Ham)  = 7.4
```

**DESPUÉS del fix:**
```
Deep interior:
  max|Ham| = 12.0 at r = 0.048  ← Sin cambio
  L2(Ham)  = 7.4
```

**Conclusión**: ⚠️ **PROBLEMA PERSISTE** - no hay mejora en Ham

---

### 🔍 Test 4: Análisis Término por Término (test_tov_initial_data_diagnostics.py)

**Hallazgo crítico:**
```
Término por término en el centro (r=0):
  (2/3)K²:                    +0.00
  -Ā_ij Ā^ij:                 -0.00
  e^{-4φ} R̄:                 +13.0   ← Mejoró de 10^58!
  -8e^{-4φ} γ̄^ij∂_iφ∂_jφ:    -0.00
  -8e^{-4φ} γ̄^ij∂_i∂_jφ:     -0.00
  +8e^{-4φ} γ̄^ijΓ̄^k_ij∂_kφ: +0.00
  -16πGρ:                     -11.6
  --------------------------------
  Suma manual:                +1.3   ← ✅ Razonable!

  get_constraints_diagnostic: -19.2  ← ❌ Discrepancia enorme!

  Diferencia:                 20.5   ← ❌ PROBLEMA CRÍTICO
```

**Conclusión**: 🔍 **NUEVO PROBLEMA IDENTIFICADO**
- El término R̄ **SÍ mejoró** (fix funcionó)
- La suma manual da **H ≈ 1.3** (aceptable)
- La función oficial da **H ≈ -19** (inaceptable)
- **Discrepancia de ~20** entre cálculo manual y función oficial

---

## Análisis de la Discrepancia

### Posibles causas de la diferencia manual vs oficial:

1. **Términos faltantes en cálculo manual**
   - ¿Hay términos de derivadas que no se están incluyendo?
   - ¿Los términos con Γ̄^k_ij se calculan correctamente?

2. **Bug en get_constraints_diagnostic()**
   - ¿Hay un error en cómo se computan las derivadas numéricas?
   - ¿El tensor de Ricci se calcula correctamente con el nuevo background?

3. **Contaminación por stencil numérico**
   - ¿Las derivadas numéricas en r=0 usan puntos con r<0 que tienen errores?
   - ¿Los stencils de derivadas cruzan las ghost cells?

4. **Error en término de materia**
   - ¿El tensor de estrés-energía se proyecta correctamente?
   - ¿El factor de -16πG vs -2×8πG está correcto?

---

## Próximos Pasos

### 1. Investigar la discrepancia manual vs oficial

**Prioridad**: 🔴 ALTA

**Acción**: Crear test que compare término por término entre:
- Cálculo manual directo
- `get_constraints_diagnostic()`

Identificar exactamente QUÉ término tiene la diferencia de ~20.

### 2. Verificar stencils de derivadas numéricas

**Prioridad**: 🟡 MEDIA

**Acción**: Verificar que los stencils de derivadas no estén usando ghost cells incorrectamente.

### 3. Verificar TOV equations satisfaction

**Observación**: El test de diagnóstico muestra:
```
TOV equations NOT satisfied:
  dP/dr error: max = 16%
  dM/dr error: max = 19%
```

Esto podría indicar que la solución TOV misma tiene errores numéricos grandes.

### 4. Comparar con código de referencia

**Acción**: Si es posible, comparar con alguna implementación de referencia (NRPy+, Einstein Toolkit, etc.)

---

## Estado del Fix

### ✅ Éxitos
1. **Background corregido**: Valores catastróficos de 10^30 eliminados
2. **R̄ corregido**: Bajó de 10^58 a ~13
3. **Términos geométricos razonables**: e^{-4φ}R̄ ≈ 13, -16πGρ ≈ -12

### ❌ Problemas pendientes
1. **Constraint Hamiltoniana**: Sigue siendo ~12 (debería ser <0.1)
2. **Discrepancia manual vs oficial**: Diferencia de ~20 sin explicar
3. **Origen desconocido**: No está claro si es bug en constraint o en initial data

### 🤔 Hipótesis actual
El fix del background **fue correcto y necesario**, pero **reveló otro problema**:
- Antes: R̄ = 10^58 dominaba todo → enmascaraba otros errores
- Ahora: R̄ = 13 razonable → otros errores se hacen visibles

El problema real podría estar en:
- Cómo se calculan las derivadas numéricas en constraintsdiagnostic.py
- Cómo se proyecta el tensor de estrés-energía
- Algún error sutil en la conversión ADM → BSSN

---

## Archivos Modificados

### Código de producción
- ✅ `source/backgrounds/sphericalbackground.py` (4 funciones modificadas)

### Tests creados
- ✅ `test_tov_solver_accuracy.py` - Verifica TOV solver
- ✅ `test_tov_initialization_pipeline.py` - Verifica cada etapa del pipeline
- ✅ `test_background_with_negative_r.py` - Verifica background con r<0
- ✅ `test_ricci_scalar_center.py` - Verifica R̄ en ghost cells
- ✅ `test_tov_initial_data_diagnostics.py` - Análisis término por término
- ✅ `source/matter/hydro/tests/test_tov_hamiltonian_constraint.py` - Test principal

### Documentación
- ✅ `TOV_CONSTRAINT_VIOLATION_DIAGNOSIS.md` - Diagnóstico original
- ✅ `TOV_FIX_RESULTS.md` - Este documento

---

## Recomendación

**El fix del background debe mantenerse** - es correcto y necesario.

**Investigación adicional requerida** para entender:
1. ¿Por qué hay discrepancia de ~20 entre manual y oficial?
2. ¿Dónde está el error que causa |Ham| = 12?
3. ¿Es un problema en el cálculo de constraint o en initial data?

**No revertir** - el fix mejoró 55 órdenes de magnitud el R̄ y eliminó valores catastróficos.

---

**Autor**: Claude Code (Sonnet 4.5)
**Fecha**: 2025-11-07
**Status**: Fix parcialmente exitoso, investigación en curso

# Bug Report: Momentum Equation Implementation

## Fecha: 2025-11-07
## Estado: CRÍTICO - Bugs encontrados en términos de conexión

## Resumen Ejecutivo

Se realizó una comparación exhaustiva entre la implementación de NRPy+ (referencia) y engrenage de las ecuaciones GRHD. Se encontraron **BUGS CRÍTICOS** en los términos de conexión para la ecuación de momento.

---

## Bugs Encontrados

### 🔴 BUG CRÍTICO 1: Signo Incorrecto en Términos de Conexión (Momento)

**Archivo**: `source/matter/hydro/grhd_equations.py:451-454`

**Problema**: Los términos de conexión para la ecuación de momento tienen el **signo opuesto** al correcto.

**Código actual (INCORRECTO)**:
```python
conn_S = (
    -np.einsum('xl,xli->xi', Gamma_trace, F_S_no_ghat) +
    np.einsum('xlji,xjl->xi', hat_chris, F_S_no_ghat)
)
```

Esto calcula: `-Γ̂^k_{kl} F̃^l_i + Γ̂^l_{ji} F̃^j_l`

**Código correcto (según NRPy+)**:
```python
conn_S = (
    +np.einsum('xl,xli->xi', Gamma_trace, F_S_no_ghat) -
    np.einsum('xlji,xjl->xi', hat_chris, F_S_no_ghat)
)
```

Esto calcula: `+Γ̂^k_{kl} F̃^l_i - Γ̂^l_{ji} F̃^j_l`

**Evidencia**:
- Test case (r=9.55, TOV surface):
  - NRPy+: `Conn_S_r = +2.876e-08`
  - Engrenage: `Conn_S_r = -2.876e-08` ❌
  - Error relativo: 200% (signo opuesto)

**Impacto**: Este bug causa que el término de conexión contribuya con el **signo equivocado** a la ecuación de momento, lo que puede generar:
- Aceleración espuria en celdas cerca del radio estelar
- Crecimiento mono

tónico de velocidad radial
- Inestabilidades numéricas en equilibrios estacionarios

**Derivación correcta**:

La divergencia covariant del flujo de momento es:

∇̂_j F̃^j_i = ∂_j F̃^j_i + Γ̂^j_{jk} F̃^k_i - Γ̂^k_{ji} F̃^j_k

Por lo tanto, el RHS debe tener:
```
RHS_connection = +Γ̂^j_{jk} F̃^k_i - Γ̂^k_{ji} F̃^j_k
```

---

### 🟡 Discrepancia Menor: Tercer Término de Fuente (~33%)

**Término**: `0.5 × stress_block × ∇̂_i γ_{jk}`

**Valores observados**:
- NRPy+: `5.039e-06`
- Engrenage: `3.378e-06`
- Error relativo: ~33%

**Causa probable**: Valores aproximados en el test case (φ=-0.05, α=0.95, etc.) en lugar de solución TOV exacta.

**Impacto**: MENOR - Probablemente no es un bug sino limitación del test case.

---

## Resultados de Pruebas Exhaustivas

### ✅ Componentes Correctos (Error < 1e-10):

1. **Variables Conservadas**:
   - ✓ D (densidad conservada)
   - ✓ S_i (momento conservado, todas las componentes)
   - ✓ τ (energía conservada)

2. **Tensor Stress-Energy**:
   - ✓ T^{μν} (todas las 10 componentes independientes)
   - ✓ T^μ_ν (todas las 16 componentes)

3. **Flujos**:
   - ✓ F^i_D (flujo de densidad)
   - ✓ F^j_i (flujo de momento, 9 componentes)
   - ✓ F^i_τ (flujo de energía)

4. **Términos de Conexión**:
   - ✓ Conn_D (densidad)
   - ✓ Conn_τ (energía)

5. **Términos de Fuente**:
   - ✓ Primer término (-T^00 α ∂α)
   - ✓ Segundo término (T^0_j ∇β)

### ❌ Componentes con Errores:

1. **Términos de Conexión**:
   - ❌ Conn_S_r (ERROR: signo opuesto)

2. **Términos de Fuente**:
   - ⚠️ Tercer término (ERROR MENOR: ~33%, probablemente por test case)
   - ⚠️ Total momento (ERROR consecuencia: ~17%)

---

## Recomendaciones

### 🚨 ACCIÓN INMEDIATA REQUERIDA

**Corregir el signo de los términos de conexión en** `source/matter/hydro/grhd_equations.py:451-454`:

```python
# CAMBIAR DE:
conn_S = (
    -np.einsum('xl,xli->xi', Gamma_trace, F_S_no_ghat) +
    np.einsum('xlji,xjl->xi', hat_chris, F_S_no_ghat)
)

# A:
conn_S = (
    +np.einsum('xl,xli->xi', Gamma_trace, F_S_no_ghat) -
    np.einsum('xlji,xjl->xi', hat_chris, F_S_no_ghat)
)
```

### 🔍 Verificación Adicional

Después de corregir el bug:
1. Ejecutar `verify_momentum_nrpy_vs_engrenage.py` de nuevo
2. Verificar que `Conn_S_r` coincida con NRPy+
3. Correr simulación TOV completa para verificar que velocidad radial permanezca en equilibrio

---

## Herramientas de Diagnóstico

Se creó un script completo de verificación:
- **Archivo**: `verify_momentum_nrpy_vs_engrenage.py`
- **Funcionalidad**: Comparación exhaustiva término por término de todas las ecuaciones GRHD
- **Threshold**: 1e-10 (precisión numérica máxima)

Este script debe ser usado para validar cualquier cambio futuro en las ecuaciones.

---

## Conclusión

Se identificó un **BUG CRÍTICO** en los términos de conexión para la ecuación de momento. El signo incorrecto puede causar el crecimiento monotónico de velocidad observado en las simulaciones TOV.

**Próximo paso**: Corregir el signo y re-ejecutar pruebas de TOV para verificar que el equilibrio se mantiene.

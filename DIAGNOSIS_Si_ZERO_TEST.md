# Resultado Test Si = 0: Velocidad SIGUE CRECIENDO

## Fecha
2025-11-07

## Test Realizado

**Hipótesis**: Feedback loop BSSN-materia causa crecimiento de v^r
```
S_i → λ^i → Γ^i_{jk} → momentum RHS → S_i crece
```

**Test**: Forzar `emtensor.Si[:] = 0.0` en [perfect_fluid.py:204](source/matter/perfect_fluid.py:204)

**Predicción**: Si el feedback es la causa, v^r debería dejar de crecer

## Resultado

### ⚠️ VELOCIDAD SIGUE CRECIENDO con Si = 0 forzado

```
step  200  t=1.29:  max_vʳ = 1.096e-02 (r=9.55)  max_Sʳ = 3.300e-07 (r=4.43)
step  400  t=2.58:  max_vʳ = 1.634e-02 (r=9.55)  max_Sʳ = 6.526e-07 (r=4.46)
step  800  t=5.16:  max_vʳ = 2.965e-02 (r=9.55)  max_Sʳ = 1.254e-06 (r=4.59)
step 1400 t=9.02:  max_vʳ = 4.569e-02 (r=9.55)  max_Sʳ = 2.048e-06 (r=4.95)
step 2800 t=18.05: max_vʳ = 7.953e-02 (r=9.55)  max_Sʳ = 4.346e-06 (r=6.40)
```

**Observaciones**:
1. v^r crece monotónicamente (~lineal en tiempo)
2. v^r_max siempre en r=9.55 (superficie estelar)
3. S^r también crece: 3.3e-07 → 4.3e-06 (13x en 2800 pasos)
4. Ubicación de S^r_max se mueve hacia afuera: r=4.43 → r=6.40

## Conclusión

**❌ HIPÓTESIS DE FEEDBACK LOOP RECHAZADA**

El crecimiento de v^r NO es causado por el acoplamiento BSSN-materia.

## ¿Qué hemos descartado?

1. ✓ **Ecuaciones de momento** - Verificadas contra NRPy+, error < 10^-15
   - Términos de conexión: ✓
   - Términos fuente: ✓
   - Factor conformal: ✓

2. ✓ **Acoplamiento BSSN lambda^i** - Verificado contra NRPy+
   - Coeficiente -16π: ✓
   - Signo: ✓
   - Índices: ✓

3. ✓ **Feedback loop Si → lambda → Christoffel** - Roto por Si=0, sigue creciendo

## ¿Dónde DEBE estar el bug entonces?

### Posibilidad 1: Condiciones Iniciales TOV
**Hipótesis**: v^r ≠ 0 en t=0, crece desde ahí

**Verificar**:
- ¿Es v^r exactamente cero en las condiciones iniciales?
- ¿La interpolación introduce velocidades espurias?
- ¿El equilibrio hidrostático se satisface numéricamente?

**Test**: Imprimir v^r, S^r en t=0 en toda la grilla

### Posibilidad 2: Ecuación de Energía (τ)
**Hipótesis**: Bug en RHS de τ → presión incorrecta → gradiente de presión → velocidad

**Verificar**:
- ¿Comparamos TODOS los términos de la ecuación de τ con NRPy+?
- ¿Hay algún término fuente en τ que no verificamos?

**Test**: Verificar RHS de τ exhaustivamente contra NRPy+

### Posibilidad 3: Cons2Prim Solver
**Hipótesis**: Cons2prim introduce velocidades espurias al recuperar primitivas

**Verificar**:
- ¿Falla cons2prim en algún punto?
- ¿Está usando atmósfera como fallback incorrectamente?
- ¿El método iterativo converge correctamente?

**Test**:
- Revisar c2p_fails (actualmente 0)
- Monitorear residuo de cons2prim
- Comparar v^r de cons2prim con valor analítico

### Posibilidad 4: Condiciones de Frontera
**Hipótesis**: Ghost cells permiten flujo hacia adentro

**Verificar**:
- ¿Qué condiciones de frontera se usan en r_max?
- ¿Son correctas para TOV estático?
- ¿Hay reflexión en la frontera?

**Test**: Verificar v^r en las ghost cells

### Posibilidad 5: Aplicación del RHS
**Hipótesis**: El RHS es correcto pero se aplica mal (integrador temporal, stencils)

**Verificar**:
- ¿El integrador RK4 se aplica correctamente?
- ¿Los stencils de diferencias finitas son correctos?
- ¿Hay algún signo cambiado al ensamblar el RHS final?

### Posibilidad 6: Término NO Verificado
**Hipótesis**: Hay algún término en las ecuaciones que NO comparamos con NRPy+

**Revisar**:
- ¿Comparamos el RHS TOTAL ensamblado?
- ¿Verificamos la divergencia de flujos?
- ¿Revisamos TODOS los términos geométricos?

## Patrón Sospechoso

**Observación clave**:
- v^r_max está en la SUPERFICIE (r=9.55)
- S^r_max está en el INTERIOR (r~5) y se mueve hacia afuera
- Esto sugiere que el error se propaga desde adentro hacia la superficie

**Interpretación posible**:
1. Pequeño error en el interior genera S^r ≠ 0
2. S^r se propaga hacia afuera (conservación de momento)
3. En la superficie (baja densidad), S^r → v^r grande

Esto apunta a:
- **Condiciones iniciales** (error inicial en S^r)
- **Ecuación de densidad/energía** (genera S^r espurio)
- **Ecuación de momento en el interior** (no solo en superficie)

## Acción Inmediata Recomendada

### 1. Verificar v^r en t=0
```python
# Imprimir condiciones iniciales
print("Initial conditions check:")
print(f"  max(v^r) at t=0: {np.max(np.abs(primitives['vr']))}")
print(f"  max(S^r) at t=0: {np.max(np.abs(matter.Sr))}")
print(f"  v^r profile at t=0:")
for i in range(0, N, 10):
    print(f"    r={grid.r[i]:6.2f}: v^r={primitives['vr'][i]:+.6e}, S^r={matter.Sr[i]:+.6e}")
```

### 2. Comparar RHS TOTAL con NRPy+
No solo términos individuales, sino el RHS final ensamblado:
```python
rhs_engrenage = matter.get_matter_rhs(...)
rhs_nrpy = compute_nrpy_rhs(...)  # Implementar
print(f"RHS difference: {np.max(np.abs(rhs_engrenage - rhs_nrpy))}")
```

### 3. Desactivar Si=0
Ahora que confirmamos que no es el problema, restaurar:
```python
# Comentar en perfect_fluid.py:204
# emtensor.Si[:] = 0.0
```

## Referencias

- [BUGS_FIXED_SUMMARY.md](BUGS_FIXED_SUMMARY.md) - Bugs anteriores corregidos
- [BSSN_LAMBDA_COMPARISON.md](BSSN_LAMBDA_COMPARISON.md) - Verificación BSSN coupling
- [verify_momentum_nrpy_vs_engrenage.py](verify_momentum_nrpy_vs_engrenage.py) - Test exhaustivo

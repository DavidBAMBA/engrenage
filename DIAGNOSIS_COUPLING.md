# Diagnóstico: Problema de Acoplamiento BSSN-Materia

## Observaciones de la Simulación

```
step  200:  max_vʳ=1.096e-02 (r=9.55)  max_Sʳ=3.300e-07 (r=4.43)
step  400:  max_vʳ=1.634e-02 (r=9.55)  max_Sʳ=6.526e-07 (r=4.46)
step  600:  max_vʳ=2.335e-02 (r=9.55)  max_Sʳ=9.617e-07 (r=4.50)
step  800:  max_vʳ=2.965e-02 (r=9.55)  max_Sʳ=1.254e-06 (r=4.59)
step 1000:  max_vʳ=3.495e-02 (r=9.55)  max_Sʳ=1.529e-06 (r=4.69)
step 1200:  max_vʳ=4.013e-02 (r=9.55)  max_Sʳ=1.792e-06 (r=4.79)
step 1400:  max_vʳ=4.569e-02 (r=9.55)  max_Sʳ=2.048e-06 (r=4.95)
```

### Patrones Observados:
1. **Velocidad crece en superficie** (r=9.55)
2. **Sr crece dentro de la estrella** (r=4-5)
3. **Ubicación de max Sr se mueve hacia fuera** con el tiempo
4. **Sr es muy pequeño** (~10^-7) pero no cero
5. **Crecimiento es lineal** con el tiempo

## Análisis

### ✓ Ecuaciones de Momento: CORRECTAS
Verificación exhaustiva contra NRPy+ muestra que TODOS los términos coinciden con precisión de máquina (~10^{-16}):
- Términos de conexión: ✓ (bug corregido)
- Términos fuente: ✓ (bug de factor conformal corregido)
- Flujos: ✓
- Tensor momento-energía: ✓

### ❌ Loop de Retroalimentación BSSN-Materia

**Hipótesis:** Ciclo vicioso en acoplamiento

```
Sr != 0 inicial (residuo numérico pequeño)
    ↓
emtensor.Si != 0 se pasa a BSSN
    ↓
BSSN: dlambda^i/dt tiene término: -2 * 8πG * alpha * gamma^ij * S_j  (línea 172)
    ↓
lambda^i cambia → Christoffel Γ̂^i_{jk} cambian
    ↓
Christoffel alterados → términos de conexión en ecuaciones momento cambian
    ↓
RHS de momento alterado → Sr crece
    ↓
VOLVER AL INICIO (retroalimentación positiva infinita)
```

### Ecuación BSSN Problemática

En `source/bssn/bssnrhs.py:172`:
```python
dlambdadt = (...
             - 2.0 * eight_pi_G * bssn_vars.lapse[:,np.newaxis] *
                     np.einsum('xij,xj->xi', bar_gamma_UU, emtensor.Si))
```

Este término acopla el **momento de la materia** directamente a la evolución de λ^i.

### Cálculo Numérico del Efecto

En superficie TOV (r=9.55):
```
rho0 = 1.28e-5
vr = 0.01 (después de growth)
Sr = 1.54e-7

Término en BSSN:
  -2 * 8πG * alpha * Sr ≈ -7.7e-6
```

Este término hace que λ^i cambie, lo que altera los Christoffel, lo que afecta las conexiones en las ecuaciones de momento, lo que hace crecer Sr más.

## Condiciones Iniciales

De `tov_initial_data_interpolated.py:405`:
```python
vr_arr = np.zeros(N)  # Static TOV: vr = 0 everywhere
```

Y luego en línea 451:
```python
D_arr, Sr_arr, tau_arr = prim_to_cons(rho_arr, vr_arr, P_arr, gamma_rr, eos)
```

Con vr=0 → Sr debería ser exactamente 0.

**Pero** errores de redondeo, interpolación, o evolución BSSN pueden crear Sr ~ 10^{-15} residual, que luego crece por el loop.

## Posibles Soluciones

### Opción 1: Forzar Sr=0 en condiciones iniciales (parche)
```python
# En tov_initial_data_interpolated.py después de prim_to_cons:
Sr_arr[:] = 0.0  # Force exactly zero momentum for static TOV
```

### Opción 2: Añadir damping a lambda^i (peligroso)
Podría introducir disipación artificial, pero puede crear otros problemas.

### Opción 3: Verificar signo de emtensor.Si
**CRÍTICO:** Verificar que el signo de S_i que se pasa a BSSN es correcto.

La proyección ADM debería ser:
```
S_i = -γ_{iμ} n_ν T^{μν}
```

¿El signo es correcto en la implementación?

### Opción 4: Revisar definición de emtensor.Si

De `stress_energy.py:264`:
```python
v_D = np.einsum('nij,nj->ni', self.geometry.gamma_LL, self.v_U)
S_D = rho_h_W2[:, np.newaxis] * v_D
```

Y luego en `perfect_fluid.py:198`:
```python
emtensor.Si = em.S_D
```

¿Este es el S_i correcto para BSSN? ¿O debería ser S^i?

## Verificación Urgente Necesaria

1. **Verificar sign/índices de emtensor.Si**
   - ¿Es S_i (lower) o S^i (upper)?
   - ¿El signo es correcto?
   - NRPy+ usa qué convención?

2. **Verificar ecuación λ^i en NRPy+**
   - ¿NRPy+ usa S_i o S^i en la ecuación de λ^i?
   - ¿El signo del término es correcto?

3. **Test simple: forzar emtensor.Si = 0 en get_emtensor**
   - Si esto elimina el crecimiento → confirma el loop de retroalimentación
   - Si aún crece → el problema está en otro lugar

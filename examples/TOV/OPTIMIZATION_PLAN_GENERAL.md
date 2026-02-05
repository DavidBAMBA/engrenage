# Plan de Optimización General - TOVEvolution.py
## Meta: 10x Speedup (Funciona para Cowling Y Dynamic Mode)

**Fecha**: 2026-02-03
**Performance Actual**: 0.670s para 100 pasos RK4 (N=400)
**Meta**: 0.067s para 100 pasos (10x más rápido)

---

## 🎯 Optimizaciones Generales (No dependen de modo Cowling)

### Fase 1: Optimizaciones Inmediatas (2-3x) ⚡

#### 1.1 Reducir Llamadas a fill_boundaries
**Impacto**: 5-8% speedup
**Esfuerzo**: Bajo (30 min)

**Problema**: `fill_boundaries` se llama 4 veces por paso RK4 (una por cada stage)

```python
# ACTUAL: 4 llamadas por paso
def get_rhs_cowling(t, y, ...):
    state = y.reshape((grid.NUM_VARS, grid.N))
    grid.fill_boundaries(state)  # ← Llamada #1
    ...

def rk4_step(state_flat, dt, ...):
    k1 = get_rhs_cowling(...)  # fill_boundaries
    state_2 = state_flat + 0.5 * dt * k1
    k2 = get_rhs_cowling(...)  # fill_boundaries
    state_3 = state_flat + 0.5 * dt * k2
    k3 = get_rhs_cowling(...)  # fill_boundaries
    state_4 = state_flat + dt * k3
    k4 = get_rhs_cowling(...)  # fill_boundaries
```

**Solución**: Llenar boundaries solo donde sea necesario

```python
# OPTIMIZADO: 1-2 llamadas por paso
def get_rhs_no_boundary_fill(t, y, ...):
    """RHS sin fill_boundaries - se llama externamente."""
    state = y.reshape((grid.NUM_VARS, grid.N))
    # NO fill boundaries aquí
    ...

def rk4_step_optimized(state_flat, dt, ...):
    state = state_flat.reshape((grid.NUM_VARS, grid.N))
    grid.fill_boundaries(state)  # Solo una vez al inicio

    k1 = get_rhs_no_boundary_fill(0, state.flatten(), ...)

    state_2 = state + 0.5 * dt * k1.reshape(...)
    # Solo llenar si los stencils lo requieren
    # grid.fill_boundaries(state_2)  # Opcional
    k2 = get_rhs_no_boundary_fill(0, state_2.flatten(), ...)

    state_3 = state + 0.5 * dt * k2.reshape(...)
    k3 = get_rhs_no_boundary_fill(0, state_3.flatten(), ...)

    state_4 = state + dt * k3.reshape(...)
    k4 = get_rhs_no_boundary_fill(0, state_4.flatten(), ...)

    state_new = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4).reshape(...)
    return state_new.flatten()
```

**Archivo a modificar**: `TOVEvolution.py:142-180`

---

#### 1.2 Cache de Operaciones Repetidas en RHS
**Impacto**: 10-15% speedup
**Esfuerzo**: Medio (2-3 horas)

**Problema**: Algunas cantidades se recalculan múltiples veces en cada RHS

Ejemplo en `valencia_reference_metric.py:122` (`_extract_geometry`):
```python
# Se llama 2 veces por RHS (800 veces total en 400 RHS)
def _extract_geometry(...):
    bar_gamma_LL = get_bar_gamma_LL(r, h_LL, background)  # ← Caro
    gamma_LL = e4phi * bar_gamma_LL
    gamma_UU = inv_3x3(gamma_LL)  # ← MUY caro (0.064s total)
    ...
```

**Solución**: Estructura de datos para geometría computada

```python
@dataclass
class GeometryState:
    """Geometría computada una vez y reutilizada."""
    gamma_LL: np.ndarray
    gamma_UU: np.ndarray
    bar_gamma_LL: np.ndarray
    bar_gamma_UU: np.ndarray
    gamma_rr: np.ndarray
    e4phi: np.ndarray
    e6phi: np.ndarray

def compute_geometry_once(bssn_vars, background, r):
    """Computa geometría una sola vez."""
    e4phi = np.exp(4.0 * bssn_vars.phi)
    e6phi = np.exp(6.0 * bssn_vars.phi)

    bar_gamma_LL = get_bar_gamma_LL(r, bssn_vars.h_LL, background)
    gamma_LL = e4phi[:, np.newaxis, np.newaxis] * bar_gamma_LL

    # Inversión solo UNA vez
    gamma_UU = inv_3x3(gamma_LL)
    bar_gamma_UU = inv_3x3(bar_gamma_LL)

    return GeometryState(
        gamma_LL=gamma_LL,
        gamma_UU=gamma_UU,
        bar_gamma_LL=bar_gamma_LL,
        bar_gamma_UU=bar_gamma_UU,
        gamma_rr=gamma_LL[:, 0, 0],
        e4phi=e4phi,
        e6phi=e6phi
    )

def compute_rhs(state, bssn_vars, bssn_d1, background, r):
    """RHS optimizado - geometría computada solo una vez."""
    # Compute geometry ONCE
    geom = compute_geometry_once(bssn_vars, background, r)

    # Pasar geometría a todas las subfunciones
    fluxes = compute_fluxes(..., geom)
    sources = compute_sources(..., geom)

    return -divergence(fluxes) + sources
```

**Archivos a modificar**:
- `valencia_reference_metric.py:122` (`_extract_geometry`)
- `valencia_reference_metric.py:412` (`compute_rhs`)

---

#### 1.3 Vectorización de Loops Residuales
**Impacto**: 3-5% speedup
**Esfuerzo**: Bajo (1 hora)

**Buscar loops no vectorizados**:
```bash
# Buscar loops Python puros (no Numba)
grep -n "for.*in range" source/matter/hydro/*.py | grep -v "@jit"
```

**Ejemplo**: Reemplazar loops elemento-por-elemento con operaciones vectorizadas

```python
# ANTES
for i in range(N):
    if rho[i] < rho_floor:
        rho[i] = rho_floor
        v[i] = 0.0

# DESPUÉS
mask = rho < rho_floor
rho[mask] = rho_floor
v[mask] = 0.0
```

---

### Fase 2: JAX Compilation (5-10x) 🚀

#### 2.1 JAX para Pipeline Completo
**Impacto**: 5-8x speedup (CPU), 10-20x (GPU)
**Esfuerzo**: Alto (1-2 semanas)

**Estrategia**: Convertir el RHS completo a JAX

**Ventajas de JAX**:
1. **XLA fusion**: Combina operaciones automáticamente
2. **Elimina overhead de Python**: Todo compilado a código nativo
3. **Mejor vectorización**: Optimizaciones que NumPy no puede hacer
4. **GPU support**: Gratis si tienes GPU

**Componentes a convertir**:

1. **Reconstruction** (MP5/WENO) → JAX
2. **Riemann solver** (HLL/HLLC) → JAX
3. **Cons2prim** (Kastaun) → JAX
4. **Flux computation** → JAX
5. **Source terms** → JAX

**Estructura**:

```python
import jax
import jax.numpy as jnp
from jax import jit

@jit
def compute_rhs_jax(state_bssn, state_hydro, bssn_d1, background_arrays):
    """
    RHS completo en JAX - compilado con XLA.

    Esta función se compila UNA VEZ y luego se ejecuta
    a velocidad de código C/Fortran.
    """
    # Extraer variables
    D, Sr, tau = state_hydro
    phi, h_LL, K, a_LL, lambda_r, lapse, shift_r = state_bssn

    # Geometría (JAX arrays, operaciones fusionadas)
    e4phi = jnp.exp(4.0 * phi)
    e6phi = jnp.exp(6.0 * phi)

    bar_gamma_LL = compute_bar_gamma_jax(h_LL, background_arrays)
    gamma_LL = e4phi[:, None, None] * bar_gamma_LL
    gamma_UU = jnp.linalg.inv(gamma_LL)  # Vectorizado automáticamente

    # Cons2prim (JAX-compatible)
    rho, v, p = cons2prim_kastaun_jax(D, Sr, tau, gamma_UU, e6phi)

    # Reconstruction (JAX-compatible)
    rho_L, rho_R = reconstruct_mp5_jax(rho)
    v_L, v_R = reconstruct_mp5_jax(v)
    p_L, p_R = reconstruct_mp5_jax(p)

    # Riemann solver (JAX-compatible)
    F_D, F_Sr, F_tau = hll_riemann_jax(
        rho_L, rho_R, v_L, v_R, p_L, p_R, gamma_LL
    )

    # Flux divergence (JAX gradient)
    dF_D = jnp.gradient(F_D, edge_order=2)
    dF_Sr = jnp.gradient(F_Sr, edge_order=2)
    dF_tau = jnp.gradient(F_tau, edge_order=2)

    # Source terms (JAX-compatible)
    S_D, S_Sr, S_tau = compute_sources_jax(
        rho, v, p, gamma_LL, gamma_UU, bssn_d1
    )

    # Assemble RHS
    rhs_D = -dF_D + S_D
    rhs_Sr = -dF_Sr + S_Sr
    rhs_tau = -dF_tau + S_tau

    return jnp.stack([rhs_D, rhs_Sr, rhs_tau])
```

**Implementación incremental**:

**Semana 1**: Convertir kernels individuales
- [ ] `reconstruction_jax.py` - Implementar MP5/WENO en JAX
- [ ] `riemann_jax.py` - Implementar HLL/HLLC en JAX
- [ ] Benchmark individual de cada kernel

**Semana 2**: Integrar pipeline completo
- [ ] `cons2prim_jax.py` - Kastaun solver en JAX
- [ ] `valencia_jax.py` - RHS completo en JAX
- [ ] Integrar con `TOVEvolution.py`

**Semana 3**: Testing y optimización
- [ ] Tests de conservación de masa
- [ ] Tests de estabilidad
- [ ] Tuning de performance

---

#### 2.2 Hybrid Approach: Numba + JAX Orchestration
**Impacto**: 3-5x speedup (más fácil que full JAX)
**Esfuerzo**: Medio (1 semana)

**Idea**: Mantener kernels Numba, usar JAX solo para orquestación

```python
import jax
import jax.numpy as jnp
from jax import jit
from jax.experimental import host_callback

# Mantener kernels Numba existentes
from source.bssn.tensoralgebra_kernels import inv_3x3  # Numba
from source.matter.hydro.riemann import solve_batch_fused  # Numba

@jit
def orchestrate_rhs_jax(state, params):
    """
    JAX orquesta el pipeline, pero llama a kernels Numba.
    """
    # JAX operations (fusionadas)
    D, Sr, tau = jnp.split(state, 3)

    # Call Numba kernel via host_callback
    gamma_UU = host_callback.call(
        inv_3x3,
        gamma_LL,
        result_shape=jax.ShapeDtypeStruct(gamma_LL.shape, gamma_LL.dtype)
    )

    # JAX operations continue
    ...

    return rhs
```

**Ventajas**:
- Menos trabajo de conversión
- Reutiliza código Numba existente
- JAX solo para fusión de operaciones de alto nivel

**Desventajas**:
- Speedup menor que full JAX (~3-5x vs ~5-8x)
- No puede ejecutar en GPU

---

### Fase 3: Advanced Optimizations (1.5-2x adicional)

#### 3.1 Fusionar Flux + Source Computation
**Impacto**: 8-12% speedup
**Esfuerzo**: Medio

**Problema**: Múltiples pasadas sobre arrays

```python
# ACTUAL: 3 pasadas
fluxes = compute_fluxes(state)       # Pasada 1
div_flux = divergence(fluxes)         # Pasada 2
sources = compute_sources(state)      # Pasada 3
rhs = -div_flux + sources             # Pasada 4
```

**Optimizado**: Una sola pasada

```python
# OPTIMIZADO: 1 pasada
rhs = compute_rhs_fused(state)  # Todo en un kernel
```

---

#### 3.2 Mixed Precision (FP32)
**Impacto**: 15-30% speedup en hardware moderno
**Esfuerzo**: Bajo

**Cambio simple**:
```python
# Usar float32 en lugar de float64
state = np.zeros((NUM_VARS, N), dtype=np.float32)
```

**Ventajas**:
- 2x menos memoria bandwidth
- Más rápido en GPUs modernas
- Más rápido en CPUs con AVX2/AVX512

**Desventajas**:
- Posible pérdida de precisión (verificar cons2prim)

---

## Roadmap de Implementación

### Sprint 1 (Semana 1): Quick Wins
- [x] Profiling completo ✅
- [ ] Reducir fill_boundaries
- [ ] Cache de geometría en RHS
- [ ] Vectorizar loops residuales
- **Meta**: 2.0-2.5x speedup

### Sprint 2 (Semanas 2-3): JAX Prototype
- [ ] Implementar kernels JAX básicos
- [ ] Benchmark individual kernels
- [ ] Implementar RHS completo en JAX
- **Meta**: 5-8x speedup total

### Sprint 3 (Semana 4): Polish & Advanced
- [ ] Fusionar operaciones
- [ ] Test mixed precision
- [ ] Optimizar stencils
- **Meta**: 10x speedup total ✅

---

## Resumen de Speedups Esperados

| Optimización | Speedup | Acumulado | Modo |
|--------------|---------|-----------|------|
| Reduce boundaries | 1.07x | 1.07x | Ambos |
| Cache geometría en RHS | 1.15x | 1.23x | Ambos |
| Vectorización | 1.03x | 1.27x | Ambos |
| **JAX RHS completo** | **5.0x** | **6.35x** | Ambos |
| Fusionar ops | 1.10x | 6.99x | Ambos |
| Mixed precision | 1.20x | 8.38x | Ambos |
| Optimized stencils | 1.15x | **9.64x** | Ambos |

**Meta de 10x: ALCANZABLE** ✅

---

## Próximos Pasos Inmediatos

### Opción A: Quick Wins (Recomendado para empezar)
```bash
# 1. Implementar reducción de fill_boundaries
#    Archivo: TOVEvolution.py
#    Tiempo: 30 min
#    Ganancia: 1.07x

# 2. Cache de geometría
#    Archivo: valencia_reference_metric.py
#    Tiempo: 2-3 horas
#    Ganancia: 1.15x adicional
```

### Opción B: JAX Prototype (Máximo impacto)
```bash
# 1. Instalar JAX
pip install jax jaxlib

# 2. Crear valencia_jax.py
#    Tiempo: 1-2 semanas
#    Ganancia: 5-8x total

# 3. Benchmark
python benchmark_jax_full.py
```

---

**¿Quieres que implemente alguna de estas optimizaciones?**

Recomiendo empezar con **Opción A** (Quick Wins) para obtener 2-2.5x de mejora en 1 día de trabajo, y luego atacar **Opción B** (JAX) para llegar al 10x completo.

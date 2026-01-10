# Plan: Diferencias Finitas de Alto Orden con HRSC

## Objetivo
Implementar un módulo **independiente** de FD-HRSC (Finite Difference High-Resolution Shock-Capturing) para comparar con el esquema de volúmenes finitos existente. Optimizado para ondas gravitacionales y QNMs en TOV.

## Diferencia Fundamental: FV vs FD-HRSC

```
Volúmenes Finitos (actual):           Diferencias Finitas HRSC (nuevo):
────────────────────────────          ─────────────────────────────────
1. Reconstruir PRIMITIVAS             1. Calcular FLUJOS F en nodos
2. Resolver Riemann → F_{i+1/2}       2. Flux splitting: F = F⁺ + F⁻
3. dU/dt = -(F_{i+1/2}-F_{i-1/2})/dx  3. WENO reconstruir F⁺, F⁻
                                      4. dU/dt = -dF/dx (diferencias finitas)
```

---

## Archivos a Crear

### Nueva estructura:
```
source/matter/hydro/fd_hrsc/
├── __init__.py
├── flux_splitting.py      # Lax-Friedrichs splitting
├── weno_flux.py           # WENO5/WENO-Z para flujos
└── fd_valencia.py         # Clase principal FD-HRSC
```

---

## Implementación Detallada

### 1. `flux_splitting.py` - Flux Splitting de Lax-Friedrichs

```python
def compute_max_wavespeed(rho0, vr, pressure, gamma_rr, alpha, beta_r, eos_gamma):
    """
    Calcula |λ|_max usando método cuadrático (reutiliza lógica de riemann.py)
    """
    # Ecuación cuadrática para eigenvalores característicos
    # a λ² + b λ + c = 0
    pass

def lax_friedrichs_split(F, U, alpha_max):
    """
    F⁺ = 0.5 * (F + alpha_max * U)  # viaja hacia derecha
    F⁻ = 0.5 * (F - alpha_max * U)  # viaja hacia izquierda
    """
    pass
```

**Archivo referencia:** `riemann.py` - método `_find_cp_cm()` para velocidades características

### 2. `weno_flux.py` - Reconstrucción WENO para Flujos

```python
@jit(nopython=True)
def weno5_flux_plus_kernel(fm2, fm1, f0, fp1, fp2):
    """
    WENO5 para F⁺ en i+1/2 (stencil sesgado IZQUIERDA)
    Pesos ideales: d0=0.1, d1=0.6, d2=0.3
    """
    pass

@jit(nopython=True)
def weno5_flux_minus_kernel(fm1, f0, fp1, fp2, fp3):
    """
    WENO5 para F⁻ en i+1/2 (stencil sesgado DERECHA)
    Pesos ideales: d0=0.1, d1=0.6, d2=0.3 (mirror)
    """
    pass

class FluxReconstructor:
    def __init__(self, method="wenoz"):  # wenoz recomendado
        pass

    def reconstruct(self, F_plus, F_minus) -> F_half:
        """Retorna flujo numérico en half-points"""
        pass
```

**Archivo referencia:** `reconstruction.py` - indicadores de suavidad β₀, β₁, β₂

### 3. `fd_valencia.py` - Clase Principal

```python
class FDValencia:
    """
    Independiente de ValenciaReferenceMetric (FV).
    Misma interfaz compute_rhs() para intercambiabilidad.
    """

    def __init__(self, atmosphere, reconstruction_method="wenoz", splitting="local"):
        self.flux_reconstructor = FluxReconstructor(method=reconstruction_method)
        # Reutiliza source/connection terms del FV
        self._fv_valencia = ValenciaReferenceMetric(atmosphere=atmosphere)

    def compute_rhs(self, D, S, tau, rho0, v_U, pressure, W, h,
                    r, bssn_vars, bssn_d1, background, spacetime_mode,
                    eos, grid, reconstructor=None, riemann_solver=None):
        """
        1. Calcular flujo físico F en cada nodo
        2. Flux splitting: F⁺, F⁻ = lax_friedrichs_split(F, U, alpha_max)
        3. WENO reconstruct: F_half = FluxReconstructor(F⁺, F⁻)
        4. Derivada: dF/dx = (F_{i+1/2} - F_{i-1/2}) / dx
        5. RHS = -dF/dx + connection_terms + source_terms
        """
        # Reutiliza _compute_source_terms y _compute_connection_terms del FV
        pass
```

**Archivos referencia:**
- `valencia_reference_metric.py` - `_compute_fluxes()`, `_compute_source_terms()`, `_compute_connection_terms()`
- `perfect_fluid.py` - interfaz con BSSN

### 4. Modificar `perfect_fluid.py` - Selector FV/FD

```python
class PerfectFluid:
    def __init__(self, ..., method="fv"):  # NUEVO parámetro
        if method == "fv":
            self.valencia = ValenciaReferenceMetric(...)
        elif method == "fd":
            from .fd_hrsc import FDValencia
            self.valencia = FDValencia(...)
```

---

## Orden de Implementación

| Fase | Tarea | Archivos |
|------|-------|----------|
| 1 | Crear estructura directorios | `fd_hrsc/__init__.py` |
| 2 | Implementar flux splitting | `fd_hrsc/flux_splitting.py` |
| 3 | Implementar WENO para flujos | `fd_hrsc/weno_flux.py` |
| 4 | Implementar clase principal | `fd_hrsc/fd_valencia.py` |
| 5 | Integrar en PerfectFluid | `perfect_fluid.py`, `__init__.py` |
| 6 | Tests de validación | `fd_hrsc/tests/` |

---

## Tests Propuestos

1. **Test unitario flux splitting:** `F⁺ + F⁻ = F`
2. **Test orden WENO5:** Error ~ O(dx⁵) para función suave
3. **Test estado uniforme:** RHS ≈ 0
4. **Test Sod shock tube:** Comparar FV vs FD
5. **Test TOV estático:** Equilibrio por 100+ tiempos de cruce
6. **Test QNM:** Comparar frecuencia con FV y literatura

---

## Ventajas Esperadas de FD-HRSC para QNMs

| Aspecto | FV (actual) | FD-HRSC |
|---------|-------------|---------|
| Disipación en regiones suaves | Mayor | **Menor** |
| Captura de shocks | Excelente | Muy bueno |
| Costo computacional | Mayor (Riemann) | **Menor** |
| Complejidad | Alta | Media |

Para ondas gravitacionales y QNMs (señales suaves), FD-HRSC debería mostrar menor disipación numérica.

---

## Archivos Críticos a Modificar/Crear

| Archivo | Acción |
|---------|--------|
| `source/matter/hydro/fd_hrsc/__init__.py` | **Crear** |
| `source/matter/hydro/fd_hrsc/flux_splitting.py` | **Crear** |
| `source/matter/hydro/fd_hrsc/weno_flux.py` | **Crear** |
| `source/matter/hydro/fd_hrsc/fd_valencia.py` | **Crear** |
| `source/matter/hydro/perfect_fluid.py` | Modificar (añadir `method` param) |
| `source/matter/hydro/__init__.py` | Modificar (exportar FDValencia) |

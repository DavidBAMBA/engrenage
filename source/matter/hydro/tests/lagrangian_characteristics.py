#!/usr/bin/env python3
"""
lagrangian_characteristics.py — Método de características Lagrangiano para flujos isentrópicos.

Implementa el método de características para resolver la evolución de ondas suaves
en hidrodinámica relativista especial usando coordenadas Lagrangianas.

Para flujos isentrópicos (entropía constante), el invariante de Riemann J₋ se conserva
a lo largo de las características, permitiendo una solución semi-analítica de alta precisión.

Referencia:
- Rezzolla & Zanotti (2013), "Relativistic Hydrodynamics", Sección 6.4.2
"""

import numpy as np
from scipy.interpolate import CubicSpline


class LagrangianGrid:
    """Grilla Lagrangiana para método de características.

    Representa un conjunto de elementos de fluido (partículas Lagrangianas)
    que se mueven con el flujo. Cada elemento conserva su masa.

    Atributos:
        q: Coordenadas materiales (etiquetas, fijas en el tiempo)
        x: Posiciones espaciales actuales (evolucionan con el flujo)
        rho: Densidad de masa en reposo
        v: Velocidad (en unidades de c=1)
        p: Presión
        J_minus: Invariante de Riemann J₋ (constante en flujo isentrópico)
        dm: Masa de cada elemento (constante)
        eos: Ecuación de estado (PolytropicEOS)
    """

    def __init__(self, q_coords, rho0, v0, p0, eos):
        """
        Inicializa grilla Lagrangiana.

        Args:
            q_coords: Coordenadas materiales (etiquetas, fijas)
            rho0: Densidad inicial
            v0: Velocidad inicial
            p0: Presión inicial
            eos: Objeto PolytropicEOS
        """
        self.q = np.asarray(q_coords, dtype=float)  # Coordenadas materiales (fijas)
        self.x = np.copy(self.q)                    # Posiciones iniciales x=q
        self.rho = np.asarray(rho0, dtype=float)
        self.v = np.asarray(v0, dtype=float)
        self.p = np.asarray(p0, dtype=float)
        self.eos = eos

        # Calcular y guardar J₋ (constante durante evolución isentrópica)
        # J₋ = arctanh(v) - (1/√(Γ-1)) arctanh(cs/√(Γ-1))
        cs = np.sqrt(eos.sound_speed_squared(self.rho))
        sqrt_gm1 = np.sqrt(eos.gamma - 1.0)
        v_safe = np.clip(self.v, -0.9999, 0.9999)
        cs_ratio = np.clip(cs / sqrt_gm1, -0.9999, 0.9999)
        self.J_minus = np.arctanh(v_safe) - (1.0 / sqrt_gm1) * np.arctanh(cs_ratio)

        # Masa inicial por partícula (conservada)
        # Para coordenadas Lagrangianas, cada partícula lleva una masa fija
        # dm_i ≈ ρ₀(q_i) W₀(q_i) Δq_i
        # donde Δq_i es el espaciamiento Lagrangiano alrededor de la partícula
        W0 = 1.0 / np.sqrt(1.0 - self.v**2)

        # Calcular espaciamiento efectivo para cada punto (usando promedios)
        dq = np.diff(self.q)  # Espaciamiento entre puntos consecutivos
        # Para cada punto interno, usar promedio de espaciamientos vecinos
        dq_effective = np.zeros_like(self.q)
        dq_effective[0] = dq[0]  # Borde izquierdo
        dq_effective[-1] = dq[-1]  # Borde derecho
        dq_effective[1:-1] = 0.5 * (dq[:-1] + dq[1:])  # Promedio para puntos internos

        self.dm = self.rho * W0 * dq_effective  # Masa por partícula (constante)


def update_from_positions(grid):
    """Actualiza ρ, v, p desde nuevas posiciones x usando conservación + J₋.

    Args:
        grid: LagrangianGrid con posiciones x actualizadas

    Física:
        Conservación de masa Lagrangiana:
            dm = ρ₀ W₀ dx₀ = ρ W dx = ρ W (∂x/∂q) dq  (constante)

        donde W = 1/√(1-v²) es el factor de Lorentz.

        Para flujo isentrópico, J₋ = constante para cada partícula:
            J₋ = arctanh(v) - (1/√(Γ-1)) arctanh(cs/√(Γ-1))

        Dado x(t), calculamos ρ(t) y v(t) simultáneamente:
        1. Calcular ∂x/∂q desde x
        2. Iterar hasta convergencia:
           - Calcular v desde J₋ = const (usando ρ actual)
           - Calcular W desde v
           - Actualizar ρ desde conservación: ρ = dm / (W × ∂x/∂q)
    """
    # Calcular ∂x/∂q usando gradiente (diferencias centradas)
    dx_dq = np.gradient(grid.x, grid.q)

    # Verificar monotonic (no crossing de características)
    if np.any(dx_dq <= 0):
        n_crossings = np.sum(dx_dq <= 0)
        print(f"  WARNING: {n_crossings} characteristic crossings detected (caustic forming!)")
        print(f"  Min dx_dq = {np.min(dx_dq):.2e}, at x = {grid.x[np.argmin(dx_dq)]:.4f}")

    # Evitar valores negativos o muy pequeños de ∂x/∂q
    # Si hay crossing, clip para evitar explosión numérica
    dx_dq = np.maximum(dx_dq, 1e-10)

    # Iteración para resolver ρ y v simultáneamente
    # Necesario porque W depende de v, pero v depende de ρ vía J₋
    rho_new = grid.rho.copy()

    for iteration in range(15):  # Newton iterations (típicamente converge en 5-10)
        # Calcular v desde J₋ = const usando ρ actual
        cs = np.sqrt(grid.eos.sound_speed_squared(rho_new))
        sqrt_gm1 = np.sqrt(grid.eos.gamma - 1.0)
        cs_ratio = np.clip(cs / sqrt_gm1, -0.9999, 0.9999)
        arg = grid.J_minus + (1.0 / sqrt_gm1) * np.arctanh(cs_ratio)
        arg = np.clip(arg, -10.0, 10.0)  # Prevenir overflow en tanh
        v_new = np.tanh(arg)

        # Calcular W
        v_new = np.clip(v_new, -0.999999, 0.999999)  # Mantener sublumínico
        W_new = 1.0 / np.sqrt(1.0 - v_new**2)

        # Actualizar ρ desde conservación: dm = ρ W (∂x/∂q) dq
        rho_updated = grid.dm / (W_new * dx_dq)

        # Verificar convergencia
        max_change = np.max(np.abs(rho_updated - rho_new) / (rho_new + 1e-30))
        if max_change < 1e-12:
            break

        rho_new = rho_updated

    # Actualizar estado
    grid.rho = rho_new
    grid.v = v_new
    grid.p = grid.eos.pressure(grid.rho)


def rk4_step(grid, dt):
    """Un paso RK4 para evolución Lagrangiana.

    Integra dx/dt = v con alta precisión usando Runge-Kutta 4to orden.

    Args:
        grid: LagrangianGrid
        dt: Paso de tiempo
    """
    x0 = grid.x.copy()

    # k1 = v(t)
    k1 = grid.v.copy()

    # k2 = v(t + dt/2) con x = x0 + dt/2 * k1
    grid.x = x0 + 0.5 * dt * k1
    update_from_positions(grid)
    k2 = grid.v.copy()

    # k3 = v(t + dt/2) con x = x0 + dt/2 * k2
    grid.x = x0 + 0.5 * dt * k2
    update_from_positions(grid)
    k3 = grid.v.copy()

    # k4 = v(t + dt) con x = x0 + dt * k3
    grid.x = x0 + dt * k3
    update_from_positions(grid)
    k4 = grid.v.copy()

    # Actualización final: x(t+dt) = x(t) + dt/6 * (k1 + 2k2 + 2k3 + k4)
    grid.x = x0 + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    update_from_positions(grid)


def evolve_lagrangian_characteristics(grid, t_final, cfl=0.01, verbose=True):
    """Evoluciona grilla Lagrangiana hasta t_final usando método de características.

    Args:
        grid: LagrangianGrid inicializada
        t_final: Tiempo final
        cfl: Factor CFL para timestep (default: 0.01 para alta precisión)
        verbose: Imprimir progreso

    Returns:
        grid evolucionada a t_final
    """
    t = 0.0
    n_steps = 0

    while t < t_final:
        # Calcular dt desde condición CFL
        # dt = CFL × (mínimo espaciamiento) / (máxima velocidad característica)
        dx_min = np.min(np.abs(np.diff(grid.x)))
        v_max = np.max(np.abs(grid.v))
        cs_max = np.max(np.sqrt(grid.eos.sound_speed_squared(grid.rho)))
        char_speed_max = v_max + cs_max  # Velocidad característica máxima

        dt = cfl * dx_min / (char_speed_max + 1e-10)

        # Ajustar último paso para llegar exactamente a t_final
        if t + dt > t_final:
            dt = t_final - t

        # Paso RK4
        rk4_step(grid, dt)

        t += dt
        n_steps += 1

        # Imprimir progreso cada 50 pasos
        if verbose and n_steps % 50 == 0:
            print(f"  Lagrangian evolution: t = {t:.4f} / {t_final:.4f}, steps = {n_steps}")

    if verbose:
        print(f"  Lagrangian evolution complete: t = {t:.4f}, total steps = {n_steps}")

    return grid


def interpolate_to_eulerian(grid, x_euler):
    """Interpola solución Lagrangiana a grilla Euleriana usando splines cúbicos.

    Args:
        grid: LagrangianGrid evolucionada
        x_euler: Array de posiciones Eulerianas donde evaluar

    Returns:
        rho_euler, v_euler, p_euler: Arrays en posiciones x_euler

    Nota:
        Usa CubicSpline con bc_type='natural' (segunda derivada = 0 en bordes)
        para interpolación suave (C² continua).
    """
    # Verificar que x está ordenado (debería estarlo si no hay crossing de características)
    if not np.all(np.diff(grid.x) > 0):
        # Si hay puntos no ordenados, ordenar antes de interpolar
        sort_idx = np.argsort(grid.x)
        x_sorted = grid.x[sort_idx]
        rho_sorted = grid.rho[sort_idx]
        v_sorted = grid.v[sort_idx]
        p_sorted = grid.p[sort_idx]
    else:
        x_sorted = grid.x
        rho_sorted = grid.rho
        v_sorted = grid.v
        p_sorted = grid.p

    # Interpolar con splines cúbicos (C² continuos)
    # bc_type='natural': segunda derivada = 0 en los bordes
    rho_interp = CubicSpline(x_sorted, rho_sorted, bc_type='natural')
    v_interp = CubicSpline(x_sorted, v_sorted, bc_type='natural')
    p_interp = CubicSpline(x_sorted, p_sorted, bc_type='natural')

    # Evaluar en grilla Euleriana
    rho_euler = rho_interp(x_euler)
    v_euler = v_interp(x_euler)
    p_euler = p_interp(x_euler)

    return rho_euler, v_euler, p_euler

#!/usr/bin/env python
"""
Herramientas de diagnóstico para identificar bugs específicos
en la implementación de hidrodinámica relativista.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'source')

from matter.hydro.eos import IdealGasEOS
from matter.hydro.reconstruction import MinmodReconstruction
from matter.hydro.riemann import HLLERiemannSolver
from source.matter.hydro.cons2prim import cons2prim
from matter.hydro.valencia_reference_metric import ValenciaReferenceMetric

# =============================================================================
# DIAGNÓSTICO 1: VERIFICAR VELOCIDADES CARACTERÍSTICAS
# =============================================================================
def diagnose_characteristic_speeds():
    """Verificar que las velocidades características son físicas."""
    print("\n" + "="*60)
    print("DIAGNÓSTICO: Velocidades Características")
    print("="*60)
    
    eos = IdealGasEOS(gamma=1.4)
    
    # Casos de prueba
    test_cases = [
        # (rho0, vr, p, nombre)
        (1.0, 0.0, 1.0, "Reposo"),
        (1.0, 0.5, 1.0, "Subsónico"),
        (1.0, 0.9, 1.0, "Relativista"),
        (0.1, 0.0, 0.01, "Baja densidad"),
        (10.0, 0.0, 100.0, "Alta presión"),
    ]
    
    for rho0, vr, p, name in test_cases:
        eps = eos.eps_from_rho_p(rho0, p)
        h = 1.0 + eps + p/rho0
        cs2 = eos.sound_speed_squared(rho0, p, eps)
        cs = np.sqrt(cs2)
        
        # Velocidades características en Minkowski
        v2 = vr**2
        denom = 1.0 - v2*cs2
        
        if denom > 0:
            lambda_plus = (vr + cs*np.sqrt(1-v2)) / denom
            lambda_minus = (vr - cs*np.sqrt(1-v2)) / denom
        else:
            lambda_plus = lambda_minus = np.nan
        
        print(f"\n{name}:")
        print(f"  ρ₀={rho0:.2f}, v={vr:.2f}, p={p:.2f}")
        print(f"  cs={cs:.3f}, cs²={cs2:.3f}")
        print(f"  λ₊={lambda_plus:.3f}, λ₋={lambda_minus:.3f}")
        
        # Verificaciones
        if np.isnan(lambda_plus) or np.isnan(lambda_minus):
            print("  ⚠️ ADVERTENCIA: Velocidades no físicas")
        elif abs(lambda_plus) > 1.0 or abs(lambda_minus) > 1.0:
            print("  ⚠️ ADVERTENCIA: Velocidades superlumínicas")
        else:
            print("  ✓ OK")

# =============================================================================
# DIAGNÓSTICO 2: ESTABILIDAD DE cons2prim
# =============================================================================
def diagnose_cons2prim_stability():
    """Probar cons2prim en casos extremos."""
    print("\n" + "="*60)
    print("DIAGNÓSTICO: Estabilidad de cons2prim")
    print("="*60)
    
    eos = IdealGasEOS(gamma=1.4)
    
    # Casos extremos
    extreme_cases = [
        # (D, Sr, tau, descripción)
        (1e-10, 0.0, 1e-10, "Casi vacío"),
        (1000.0, 0.0, 10000.0, "Alta densidad"),
        (1.0, 0.99, 10.0, "Alto momento"),
        (1.0, 1e-10, 0.1, "Bajo momento"),
        (1.0, 0.0, -0.5, "tau negativo (no físico)"),
    ]
    
    for D, Sr, tau, desc in extreme_cases:
        print(f"\n{desc}: D={D:.2e}, Sr={Sr:.2e}, tau={tau:.2e}")
        
        try:
            result = cons2prim(
                ([D], [Sr], [tau]), eos,
                metric=(np.ones(1), np.zeros(1), np.ones(1))
            )
            
            if result['success'][0]:
                print(f"  ✓ Convergió")
                print(f"    ρ₀={result['rho0'][0]:.2e}")
                print(f"    vr={result['vr'][0]:.3f}")
                print(f"    p={result['p'][0]:.2e}")
                print(f"    W={result['W'][0]:.3f}")
                
                # Verificar consistencia
                W = result['W'][0]
                h = result['h'][0]
                rho0 = result['rho0'][0]
                vr = result['vr'][0]
                p = result['p'][0]
                
                D_check = rho0 * W
                Sr_check = rho0 * h * W**2 * vr
                tau_check = rho0 * h * W**2 - p - D_check
                
                D_error = abs(D_check - D) / (abs(D) + 1e-10)
                Sr_error = abs(Sr_check - Sr) / (abs(Sr) + 1e-10)
                tau_error = abs(tau_check - tau) / (abs(tau) + 1e-10)
                
                if max(D_error, Sr_error, tau_error) > 1e-6:
                    print(f"  ⚠️ Error de consistencia: D={D_error:.2e}, Sr={Sr_error:.2e}, tau={tau_error:.2e}")
            else:
                print(f"  ✗ No convergió - aplicó atmósfera")
                
        except Exception as e:
            print(f"  ✗ Excepción: {e}")

# =============================================================================
# DIAGNÓSTICO 3: FLUJOS NUMÉRICOS
# =============================================================================
def diagnose_numerical_fluxes():
    """Verificar consistencia de flujos numéricos."""
    print("\n" + "="*60)
    print("DIAGNÓSTICO: Flujos Numéricos HLLE")
    print("="*60)
    
    eos = IdealGasEOS(gamma=1.4)
    solver = HLLERiemannSolver()
    
    # Test 1: Estados idénticos -> flujo debe ser F = U * v
    print("\nTest 1: Estados idénticos")
    rho0 = 1.0
    vr = 0.2
    p = 0.1
    
    eps = eos.eps_from_rho_p(rho0, p)
    h = 1.0 + eps + p/rho0
    W = 1.0 / np.sqrt(1 - vr**2)
    
    D = rho0 * W
    Sr = rho0 * h * W**2 * vr
    tau = rho0 * h * W**2 - p - D
    
    U = (D, Sr, tau)
    prim = (rho0, vr, p)
    
    F = solver.solve(U, U, prim, prim, 1.0, 1.0, 0.0, eos)
    
    # Flujo esperado
    F_expected = np.array([D*vr, Sr*vr + p, (tau+p)*vr])
    
    error = np.abs(F - F_expected)
    print(f"  Flujo calculado: [{F[0]:.3e}, {F[1]:.3e}, {F[2]:.3e}]")
    print(f"  Flujo esperado:  [{F_expected[0]:.3e}, {F_expected[1]:.3e}, {F_expected[2]:.3e}]")
    print(f"  Error: {np.max(error):.2e}")
    print(f"  {'✓ OK' if np.max(error) < 1e-10 else '✗ FALLA'}")
    
    # Test 2: Discontinuidad estacionaria
    print("\nTest 2: Discontinuidad estacionaria (v=0)")
    rhoL, vrL, pL = 1.0, 0.0, 1.0
    rhoR, vrR, pR = 0.125, 0.0, 0.1
    
    epsL = eos.eps_from_rho_p(rhoL, pL)
    epsR = eos.eps_from_rho_p(rhoR, pR)
    hL = 1.0 + epsL + pL/rhoL
    hR = 1.0 + epsR + pR/rhoR
    
    DL = rhoL
    DR = rhoR
    SrL = SrR = 0.0
    tauL = rhoL * hL - pL - DL
    tauR = rhoR * hR - pR - DR
    
    F = solver.solve(
        (DL, SrL, tauL), (DR, SrR, tauR),
        (rhoL, vrL, pL), (rhoR, vrR, pR),
        1.0, 1.0, 0.0, eos
    )
    
    print(f"  Flujo: [{F[0]:.3e}, {F[1]:.3e}, {F[2]:.3e}]")
    print(f"  F_Sr (momentum) = {F[1]:.3f} (debe ser entre {pR:.3f} y {pL:.3f})")
    
    if pR <= F[1] <= pL:
        print("  ✓ OK - Flujo de momento en rango físico")
    else:
        print("  ✗ FALLA - Flujo fuera de rango")

# =============================================================================
# DIAGNÓSTICO 4: TÉRMINOS GEOMÉTRICOS
# =============================================================================
def diagnose_geometric_terms():
    """Verificar términos geométricos en coordenadas esféricas."""
    print("\n" + "="*60)
    print("DIAGNÓSTICO: Términos Geométricos Esféricos")
    print("="*60)
    
    N = 50
    r = np.linspace(0.1, 1.0, N)
    
    # Estado de prueba
    rho0 = np.ones(N)
    vr = 0.1 * np.ones(N)
    pressure = 0.1 * np.ones(N)
    
    print("\nTérmino fuente de momento: 2p/r")
    source_Sr = 2.0 * pressure / r
    
    print(f"  En r=0.1: {source_Sr[0]:.3f}")
    print(f"  En r=0.5: {source_Sr[N//2]:.3f}")
    print(f"  En r=1.0: {source_Sr[-1]:.3f}")
    
    # Verificar regularización en r→0
    print("\nRegularización cerca del origen:")
    r_small = np.array([1e-5, 1e-4, 1e-3, 1e-2, 0.1])
    p_test = 0.1
    
    for r_val in r_small:
        # Término directo
        direct = 2.0 * p_test / r_val
        
        # Término regularizado (L'Hôpital)
        if r_val < 1e-10:
            regularized = 0  # dp/dr en r=0 para p constante
        else:
            regularized = direct
        
        print(f"  r={r_val:.1e}: directo={direct:.1e}, regularizado={regularized:.1e}")

# =============================================================================
# DIAGNÓSTICO 5: COMPARACIÓN CON SOLUCIÓN ANALÍTICA
# =============================================================================
def diagnose_analytical_comparison():
    """Comparar con soluciones analíticas conocidas."""
    print("\n" + "="*60)
    print("DIAGNÓSTICO: Comparación con Soluciones Analíticas")
    print("="*60)
    
    # Test: Onda sonora linealizada
    print("\nOnda sonora linealizada (pequeña amplitud):")
    
    N = 100
    r = np.linspace(0.1, 1.0, N)
    
    # Perturbación sinusoidal pequeña
    epsilon = 0.01
    k = 2 * np.pi  # Número de onda
    
    rho0 = 1.0 + epsilon * np.sin(k * r)
    p0 = 0.1
    pressure = p0 * (rho0 / 1.0) ** 1.4  # Adiabático
    
    eos = IdealGasEOS(gamma=1.4)
    cs2 = eos.sound_speed_squared(1.0, p0, eos.eps_from_rho_p(1.0, p0))
    cs = np.sqrt(cs2)
    
    # Velocidad esperada (relación linealizada)
    vr_expected = epsilon * cs * np.sin(k * r)
    
    print(f"  Velocidad del sonido: cs = {cs:.3f}")
    print(f"  Amplitud de densidad: ε = {epsilon}")
    print(f"  Amplitud de velocidad esperada: ε*cs = {epsilon*cs:.3f}")
    
    # Verificar relación
    rho_pert = rho0 - 1.0
    v_to_rho_ratio = np.max(np.abs(vr_expected)) / np.max(np.abs(rho_pert))
    
    print(f"  Razón |v|/|δρ| = {v_to_rho_ratio:.3f} (debe ser ≈ {cs:.3f})")
    
    if abs(v_to_rho_ratio - cs) / cs < 0.1:
        print("  ✓ OK - Consistente con teoría lineal")
    else:
        print("  ⚠️ Desviación de teoría lineal")

# =============================================================================
# MAIN - EJECUTAR DIAGNÓSTICOS
# =============================================================================
if __name__ == "__main__":
    print("="*60)
    print("DIAGNÓSTICOS DETALLADOS - HIDRODINÁMICA RELATIVISTA")
    print("="*60)
    
    # Ejecutar todos los diagnósticos
    diagnose_characteristic_speeds()
    diagnose_cons2prim_stability()
    diagnose_numerical_fluxes()
    diagnose_geometric_terms()
    diagnose_analytical_comparison()
    
    print("\n" + "="*60)
    print("DIAGNÓSTICOS COMPLETADOS")
    print("="*60)
    print("\nRevisa los resultados arriba para identificar posibles problemas.")
    print("\nSugerencias de debugging:")
    print("1. Si cons2prim falla: revisar floors y casos extremos")
    print("2. Si flujos incorrectos: verificar cálculo de velocidades características")
    print("3. Si inestabilidad en r→0: revisar regularización de términos 1/r")
    print("4. Si no conserva: verificar términos geométricos y condiciones de frontera")
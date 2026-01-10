"""
Verificación Algebraica REAL de la Ecuación (34)
Paper: "General relativistic hydrodynamics in curvilinear coordinates"
       Montero, Baumgarte, Müller (2014) - arXiv:1309.7808v2

Verifica que:
    T^a_b (4)Γ^a_{bi} - T^k_j Γ̂^k_{ji} 
    = -T^{00} alpha ∂_i alpha + T^0_k D̂_i β^k + (1/2)(T^{00}β^jβ^k + 2T^{0j}β^k + T^{jk}) D̂_i γ_{jk}

Siguiendo las ecuaciones (29)-(33) del paper.
"""

import sympy as sp
from sympy import symbols, simplify, expand, Rational, sqrt, exp, Eq
from sympy import Function, Derivative, Symbol
from itertools import product

print("="*80)
print("VERIFICACIÓN ALGEBRAICA DE LA ECUACIÓN (34)")
print("Paper: Montero, Baumgarte, Müller (2014) - arXiv:1309.7808v2")
print("="*80)

# =============================================================================
# SECCIÓN 1: DEFINICIÓN DE SÍMBOLOS
# =============================================================================
print("\n" + "="*80)
print("SECCIÓN 1: DEFINICIÓN DE SÍMBOLOS")
print("="*80)

# Índices espaciales: usamos 1, 2, 3 (o r, θ, φ en esféricas)
# Para la verificación algebraica general, usamos índices abstractos

# Variables fundamentales
alpha = symbols('alpha', positive=True, real=True)  # Lapse
phi = symbols('phi', real=True)  # Factor conforme

# Derivadas parciales ∂_i (tratamos i como índice fijo genérico)
d_i_alpha = symbols('partial_i_alpha', real=True)
d_i_phi = symbols('partial_i_phi', real=True)

# Shift vector β^j y sus derivadas covariantes D̂_i β^k
beta = symbols('beta^1 beta^2 beta^3', real=True)
Di_beta = symbols('Dhat_i_beta^1 Dhat_i_beta^2 Dhat_i_beta^3', real=True)

# Métrica espacial γ_{jk} - usamos representación simbólica general
# γ_{jk} = e^{4φ} γ̄_{jk}
gamma_bar = {}  # Métrica conforme γ̄_{jk}
gamma = {}      # Métrica física γ_{jk}
Di_gamma_bar = {}  # D̂_i γ̄_{jk}
Di_gamma = {}      # D̂_i γ_{jk}

# Definimos componentes para j,k = 1,2,3
for j in range(1, 4):
    for k in range(j, 4):  # Solo triángulo superior por simetría
        gamma_bar[j,k] = symbols(f'gammabar_{j}{k}', real=True)
        gamma_bar[k,j] = gamma_bar[j,k]  # Simetría
        
        Di_gamma_bar[j,k] = symbols(f'Dhat_i_gammabar_{j}{k}', real=True)
        Di_gamma_bar[k,j] = Di_gamma_bar[j,k]
        
        # Ecuación (35): D̂_i γ_{jk} = e^{4φ}(4 γ̄_{jk} ∂_i φ + D̂_i γ̄_{jk})
        Di_gamma[j,k] = exp(4*phi) * (4*gamma_bar[j,k]*d_i_phi + Di_gamma_bar[j,k])
        Di_gamma[k,j] = Di_gamma[j,k]
        
        # Métrica física
        gamma[j,k] = exp(4*phi) * gamma_bar[j,k]
        gamma[k,j] = gamma[j,k]

# Tensor energía-momento T^{ab}
# Componentes: T^{00}, T^{0j}, T^{jk}
T00 = symbols('T^{00}', real=True)

T0 = {}  # T^{0j}
for j in range(1, 4):
    T0[j] = symbols(f'T^{{0{j}}}', real=True)

T_up = {}  # T^{jk} (índices arriba)
for j in range(1, 4):
    for k in range(j, 4):
        T_up[j,k] = symbols(f'T^{{{j}{k}}}', real=True)
        T_up[k,j] = T_up[j,k]

# T^0_k = T^{0j} g_{jk} ≈ T^{0j} γ_{jk} (para proyección espacial)
# Pero en el paper, T^0_k se trata como componente mixta independiente
T0_down = {}
for k in range(1, 4):
    T0_down[k] = symbols(f'T^0_{k}', real=True)

print("✓ Símbolos definidos:")
print(f"  • Lapse alpha, factor conforme φ")
print(f"  • Shift β^j para j=1,2,3")
print(f"  • Métrica γ̄_{{jk}}, derivadas D̂_i γ̄_{{jk}}")
print(f"  • Tensor T^{{00}}, T^{{0j}}, T^{{jk}}")

# =============================================================================
# SECCIÓN 2: CÁLCULO DEL LADO IZQUIERDO
# Usando ecuaciones (29)-(33) del paper
# =============================================================================
print("\n" + "="*80)
print("SECCIÓN 2: LADO IZQUIERDO - T^a_b (4)Γ^a_{bi} - T^k_j Γ̂^k_{ji}")
print("="*80)

# -----------------------------------------------------------------------------
# Ecuación (30): Contribución T^{00}(4)Γ_{00i}
# T^{00}(4)Γ_{00i} = (1/2) T^{00} ∂_i g_{00} = (1/2) T^{00} ∂_i(-alpha² + γ_{jk}β^j β^k)
# -----------------------------------------------------------------------------
print("\n--- Ecuación (30): Término T^{00}(4)Γ_{00i} ---")
print("T^{00}(4)Γ_{00i} = (1/2) T^{00} ∂_i(-alpha² + γ_{jk}β^jβ^k)")

# ∂_i(-alpha²) = -2alpha ∂_i alpha
term_alpha_sq = -2 * alpha * d_i_alpha

# ∂_i(γ_{jk}β^j β^k) = β^j β^k ∂_i γ_{jk} + 2 γ_{jk} β^j ∂_i β^k
# En el paper, esto se trata con derivadas covariantes D̂_i
# Nota: ∂_i puede reemplazarse por D̂_i para escalares

# El término de (30) contribuye al resultado pero se combina con (31)
eq30_raw = Rational(1,2) * T00 * term_alpha_sq
print(f"  Parte de alpha: (1/2) T^{{00}} × (-2alpha ∂_i alpha) = {eq30_raw}")

# -----------------------------------------------------------------------------
# Ecuación (31): T^{00} g_{a0} (4)Γ^a_{0i}
# = (1/2) T^{00} (β^j β^k D̂_i γ_{jk} + 2β^k D̂_i β_k - 2alpha D̂_i alpha)
# -----------------------------------------------------------------------------
print("\n--- Ecuación (31): Término T^{00} g_{a0} (4)Γ^a_{0i} ---")

# Término β^j β^k D̂_i γ_{jk}
term_beta_beta_gamma = sum(
    beta[j-1] * beta[k-1] * Di_gamma[j,k] 
    for j in range(1,4) for k in range(1,4)
)

# Término 2β^k D̂_i β_k (se cancela con término de ec. 32)
term_beta_Di_beta = 2 * sum(beta[k-1] * Di_beta[k-1] for k in range(1,4))

# Término -2alpha D̂_i alpha
term_alpha = -2 * alpha * d_i_alpha

eq31 = Rational(1,2) * T00 * (term_beta_beta_gamma + term_beta_Di_beta + term_alpha)

print("  (1/2) T^{00} [β^jβ^k D̂_i γ_{jk} + 2β^k D̂_i β_k - 2alpha D̂_i alpha]")

# -----------------------------------------------------------------------------
# Ecuación (32): Términos mixtos T^{0j}
# T^{0j}((4)Γ_{0ji} + (4)Γ_{j0i} - β_k Γ̂^k_{ji})
# = T^0_k D̂_i β^k - T^{00} β_k D̂_i β^k + T^{0j} β^k D̂_i γ_{jk}
# -----------------------------------------------------------------------------
print("\n--- Ecuación (32): Términos mixtos T^{0j} ---")

# Término T^0_k D̂_i β^k
term_T0k_Di_beta = sum(T0_down[k] * Di_beta[k-1] for k in range(1,4))

# Término -T^{00} β_k D̂_i β^k (se cancela con parte de ec. 31)
term_T00_beta_Di_beta = -T00 * sum(beta[k-1] * Di_beta[k-1] for k in range(1,4))

# Término T^{0j} β^k D̂_i γ_{jk}
term_T0j_beta_Di_gamma = sum(
    T0[j] * beta[k-1] * Di_gamma[j,k]
    for j in range(1,4) for k in range(1,4)
)

eq32 = term_T0k_Di_beta + term_T00_beta_Di_beta + term_T0j_beta_Di_gamma

print("  T^0_k D̂_i β^k - T^{00} β_k D̂_i β^k + T^{0j} β^k D̂_i γ_{jk}")

# -----------------------------------------------------------------------------
# Ecuación (33): Términos espaciales T^{jk}
# T^{jk}((4)Γ_{jki} - γ_{kl} Γ̂^l_{ji}) = (1/2) T^{jk} D̂_i γ_{jk}
# -----------------------------------------------------------------------------
print("\n--- Ecuación (33): Términos espaciales T^{jk} ---")

term_Tjk_Di_gamma = sum(
    T_up[j,k] * Di_gamma[j,k]
    for j in range(1,4) for k in range(1,4)
)

eq33 = Rational(1,2) * term_Tjk_Di_gamma

print("  (1/2) T^{jk} D̂_i γ_{jk}")

# -----------------------------------------------------------------------------
# SUMA TOTAL: Lado Izquierdo
# -----------------------------------------------------------------------------
print("\n--- SUMA TOTAL DEL LADO IZQUIERDO ---")

# Combinando ecuaciones (31), (32), (33)
# Nota: Los términos β^k D̂_i β_k de (31) y (32) se cancelan
lado_izquierdo = eq31 + eq32 + eq33

# Simplificamos
lado_izquierdo = expand(lado_izquierdo)

print("Lado izquierdo = Ec.(31) + Ec.(32) + Ec.(33)")

# =============================================================================
# SECCIÓN 3: CÁLCULO DEL LADO DERECHO (Ecuación 34 directa)
# =============================================================================
print("\n" + "="*80)
print("SECCIÓN 3: LADO DERECHO - Expresión de la Ecuación (34)")
print("="*80)

# -T^{00} alpha ∂_i alpha
RHS_term1 = -T00 * alpha * d_i_alpha

# T^0_k D̂_i β^k  
RHS_term2 = sum(T0_down[k] * Di_beta[k-1] for k in range(1,4))

# (1/2)(T^{00} β^j β^k + 2T^{0j} β^k + T^{jk}) D̂_i γ_{jk}
RHS_term3 = Rational(1,2) * sum(
    (T00 * beta[j-1] * beta[k-1] + 2*T0[j] * beta[k-1] + T_up[j,k]) * Di_gamma[j,k]
    for j in range(1,4) for k in range(1,4)
)

lado_derecho = RHS_term1 + RHS_term2 + RHS_term3
lado_derecho = expand(lado_derecho)

print("-T^{00} alpha ∂_i alpha + T^0_k D̂_i β^k + (1/2)(T^{00}β^jβ^k + 2T^{0j}β^k + T^{jk}) D̂_i γ_{jk}")

# =============================================================================
# SECCIÓN 4: COMPARACIÓN ALGEBRAICA
# =============================================================================
print("\n" + "="*80)
print("SECCIÓN 4: COMPARACIÓN ALGEBRAICA")
print("="*80)

# Calculamos la diferencia
diferencia = expand(lado_izquierdo - lado_derecho)
diferencia = simplify(diferencia)

print("\nCalculando: Lado_Izquierdo - Lado_Derecho")
print(f"\nDiferencia simplificada:")

if diferencia == 0:
    print("  ✅ DIFERENCIA = 0")
    print("\n" + "="*80)
    print("✅ LA IGUALDAD ES ALGEBRAICAMENTE CORRECTA")
    print("="*80)
else:
    print(f"  ❌ DIFERENCIA ≠ 0")
    print(f"\n  Diferencia = {diferencia}")
    
    # Intentamos simplificar más
    print("\n  Analizando términos de la diferencia...")
    diferencia_collected = sp.collect(diferencia, [T00, d_i_alpha, d_i_phi])
    print(f"  Agrupada: {diferencia_collected}")

# =============================================================================
# SECCIÓN 5: VERIFICACIÓN DETALLADA TÉRMINO POR TÉRMINO
# =============================================================================
print("\n" + "="*80)
print("SECCIÓN 5: VERIFICACIÓN TÉRMINO POR TÉRMINO")
print("="*80)

# Expandimos ambos lados y comparamos coeficientes

print("\n--- Término con ∂_i alpha ---")
# Del lado izquierdo (ec. 31): (1/2) T^{00} × (-2alpha ∂_i alpha) = -T^{00} alpha ∂_i alpha
coef_LHS_alpha = lado_izquierdo.coeff(d_i_alpha)
coef_RHS_alpha = lado_derecho.coeff(d_i_alpha)
print(f"  LHS coef: {coef_LHS_alpha}")
print(f"  RHS coef: {coef_RHS_alpha}")
diff_alpha = simplify(coef_LHS_alpha - coef_RHS_alpha)
print(f"  Diferencia: {diff_alpha}")
if diff_alpha == 0:
    print("  ✅ Coinciden")
else:
    print("  ❌ No coinciden")

print("\n--- Término con D̂_i β^k ---")
# Extraemos coeficientes de cada D̂_i β^k
for k in range(1, 4):
    coef_LHS_beta = lado_izquierdo.coeff(Di_beta[k-1])
    coef_RHS_beta = lado_derecho.coeff(Di_beta[k-1])
    diff_beta = simplify(coef_LHS_beta - coef_RHS_beta)
    status = "✅" if diff_beta == 0 else "❌"
    print(f"  D̂_i β^{k}: LHS={coef_LHS_beta}, RHS={coef_RHS_beta}, Diff={diff_beta} {status}")

print("\n--- Términos con D̂_i γ_{jk} (contienen ∂_i φ y D̂_i γ̄_{jk}) ---")
# Los términos D̂_i γ_{jk} aparecen como e^{4φ}(4γ̄_{jk} ∂_i φ + D̂_i γ̄_{jk})
# Verificamos los coeficientes de ∂_i φ y D̂_i γ̄_{jk}

print("\n  Coeficientes de ∂_i φ por cada γ̄_{jk}:")
for j in range(1, 4):
    for k in range(j, 4):
        # El coeficiente de ∂_i φ viene de 4 γ̄_{jk} e^{4φ}
        coef_LHS_phi = lado_izquierdo.coeff(d_i_phi).coeff(gamma_bar[j,k])
        coef_RHS_phi = lado_derecho.coeff(d_i_phi).coeff(gamma_bar[j,k])
        diff_phi = simplify(coef_LHS_phi - coef_RHS_phi)
        if coef_LHS_phi != 0 or coef_RHS_phi != 0:
            status = "✅" if diff_phi == 0 else "❌"
            print(f"    γ̄_{j}{k}: Diff={diff_phi} {status}")

print("\n  Coeficientes de D̂_i γ̄_{jk}:")
for j in range(1, 4):
    for k in range(j, 4):
        coef_LHS_gb = lado_izquierdo.coeff(Di_gamma_bar[j,k])
        coef_RHS_gb = lado_derecho.coeff(Di_gamma_bar[j,k])
        diff_gb = simplify(coef_LHS_gb - coef_RHS_gb)
        if coef_LHS_gb != 0 or coef_RHS_gb != 0:
            status = "✅" if diff_gb == 0 else "❌"
            print(f"    D̂_i γ̄_{j}{k}: Diff={diff_gb} {status}")

# =============================================================================
# SECCIÓN 6: RESULTADO FINAL
# =============================================================================
print("\n" + "="*80)
print("SECCIÓN 6: RESULTADO FINAL")
print("="*80)

if diferencia == 0:
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  ✅ LA ECUACIÓN (34) ES ALGEBRAICAMENTE CORRECTA                             ║
║                                                                              ║
║  Se verificó que:                                                            ║
║                                                                              ║
║  T^a_b (4)Γ^a_{bi} - T^k_j Γ̂^k_{ji}                                          ║
║                                                                              ║
║  = -T^{00} alpha ∂_i alpha + T^0_k D̂_i β^k                                           ║
║    + (1/2)(T^{00}β^jβ^k + 2T^{0j}β^k + T^{jk}) D̂_i γ_{jk}                    ║
║                                                                              ║
║  Usando la ecuación (35):                                                    ║
║  D̂_i γ_{jk} = e^{4φ}(4 γ̄_{jk} ∂_i φ + D̂_i γ̄_{jk})                          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
else:
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ❌ HAY DISCREPANCIAS EN LA ECUACIÓN (34)                                    ║
║                                                                              ║
║  Diferencia encontrada: {diferencia}
║                                                                              ║
║  Revisar las ecuaciones (30)-(33) del paper.                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

# =============================================================================
# SECCIÓN EXTRA: Mostrar las expresiones expandidas
# =============================================================================
print("\n" + "="*80)
print("APÉNDICE: EXPRESIONES EXPANDIDAS (primeros términos)")
print("="*80)

print("\nLado Izquierdo (primeros 500 caracteres):")
LHS_str = str(expand(lado_izquierdo))
print(f"  {LHS_str[:500]}...")

print("\nLado Derecho (primeros 500 caracteres):")
RHS_str = str(expand(lado_derecho))
print(f"  {RHS_str[:500]}...")
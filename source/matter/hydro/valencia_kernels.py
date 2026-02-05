# valencia_kernels.py
"""
Numba JIT-compiled kernels for Valencia GRHD formulation.

These kernels replace numpy einsum operations with explicit loops
that can be parallelized with Numba's prange for better performance
on small to medium sized arrays (N ~ 1000-10000).

All kernels use:
- nopython=True: Full Numba compilation
- cache=True: Cache compiled code
- fastmath=True: Allow fast math optimizations
- parallel=True: Enable prange parallelization
"""

import numpy as np
from numba import jit, prange

SPACEDIM = 3


# ==============================================================================
# STRESS-ENERGY TENSOR KERNELS
# ==============================================================================

@jit(nopython=True, cache=True, fastmath=True, parallel=True)
def compute_T4UU_kernel(rho0, v_U, pressure, W, h, alpha, beta_U, gamma_UU,
                        T00_out, T0i_out, Tij_out):
    """
    Compute contravariant stress-energy tensor T^{μν} = ρ₀ h u^μ u^ν + P g^{μν}

    Args:
        rho0, pressure, W, h: (N,) primitive quantities
        v_U: (N, 3) spatial velocity
        alpha: (N,) lapse
        beta_U: (N, 3) shift vector
        gamma_UU: (N, 3, 3) inverse spatial metric
        T00_out: (N,) output T^{00}
        T0i_out: (N, 3) output T^{0i}
        Tij_out: (N, 3, 3) output T^{ij}
    """
    N = rho0.shape[0]

    for m in prange(N):
        alph = alpha[m]
        alph_sq = alph * alph
        rho_h = rho0[m] * h[m]
        p = pressure[m]
        Wm = W[m]

        # 4-velocity: u^0 = W/α, u^i = W(v^i - β^i/α)
        u0 = Wm / alph
        ui = np.empty(SPACEDIM)
        for i in range(SPACEDIM):
            ui[i] = Wm * (v_U[m, i] - beta_U[m, i] / alph)

        # Inverse 4-metric components
        # g^{00} = -1/α²
        g4UU_00 = -1.0 / alph_sq
        # g^{0i} = β^i/α²
        g4UU_0i = np.empty(SPACEDIM)
        for i in range(SPACEDIM):
            g4UU_0i[i] = beta_U[m, i] / alph_sq

        # g^{ij} = γ^{ij} - β^i β^j/α²
        g4UU_ij = np.empty((SPACEDIM, SPACEDIM))
        for i in range(SPACEDIM):
            for j in range(SPACEDIM):
                g4UU_ij[i, j] = gamma_UU[m, i, j] - beta_U[m, i] * beta_U[m, j] / alph_sq

        # T^{00} = ρ₀ h u^0 u^0 + P g^{00}
        T00_out[m] = rho_h * u0 * u0 + p * g4UU_00

        # T^{0i} = ρ₀ h u^0 u^i + P g^{0i}
        for i in range(SPACEDIM):
            T0i_out[m, i] = rho_h * u0 * ui[i] + p * g4UU_0i[i]

        # T^{ij} = ρ₀ h u^i u^j + P g^{ij}
        for i in range(SPACEDIM):
            for j in range(SPACEDIM):
                Tij_out[m, i, j] = rho_h * ui[i] * ui[j] + p * g4UU_ij[i, j]


@jit(nopython=True, cache=True, fastmath=True, parallel=True)
def compute_T4UD_kernel(T00, T0i, Tij, alpha, beta_U, gamma_LL,
                        T0_0_out, T0_j_out, Ti_j_out):
    """
    Compute mixed stress-energy tensor T^μ_ν = T^{μδ} g_{δν}

    Args:
        T00: (N,) T^{00}
        T0i: (N, 3) T^{0i}
        Tij: (N, 3, 3) T^{ij}
        alpha: (N,) lapse
        beta_U: (N, 3) shift vector
        gamma_LL: (N, 3, 3) spatial metric
        T0_0_out: (N,) output T^0_0
        T0_j_out: (N, 3) output T^0_j
        Ti_j_out: (N, 3, 3) output T^i_j
    """
    N = T00.shape[0]

    for m in prange(N):
        alph = alpha[m]
        alph_sq = alph * alph

        # Covariant 4-metric components
        # g_{00} = -α² + β_k β^k
        beta_sq = 0.0
        for k in range(SPACEDIM):
            beta_k = 0.0
            for l in range(SPACEDIM):
                beta_k += gamma_LL[m, k, l] * beta_U[m, l]
            beta_sq += beta_U[m, k] * beta_k
        g4DD_00 = -alph_sq + beta_sq

        # g_{0i} = β_i = γ_{ij} β^j
        g4DD_0i = np.empty(SPACEDIM)
        for i in range(SPACEDIM):
            g4DD_0i[i] = 0.0
            for j in range(SPACEDIM):
                g4DD_0i[i] += gamma_LL[m, i, j] * beta_U[m, j]

        # g_{ij} = γ_{ij}
        # (already have gamma_LL)

        # T^0_0 = T^{00} g_{00} + T^{0i} g_{i0}
        T0_0_out[m] = T00[m] * g4DD_00
        for i in range(SPACEDIM):
            T0_0_out[m] += T0i[m, i] * g4DD_0i[i]

        # T^0_j = T^{00} g_{0j} + T^{0i} g_{ij}
        for j in range(SPACEDIM):
            T0_j_out[m, j] = T00[m] * g4DD_0i[j]
            for i in range(SPACEDIM):
                T0_j_out[m, j] += T0i[m, i] * gamma_LL[m, i, j]

        # T^i_j = T^{i0} g_{0j} + T^{ik} g_{kj}
        for i in range(SPACEDIM):
            for j in range(SPACEDIM):
                Ti_j_out[m, i, j] = T0i[m, i] * g4DD_0i[j]
                for k in range(SPACEDIM):
                    Ti_j_out[m, i, j] += Tij[m, i, k] * gamma_LL[m, k, j]


# ==============================================================================
# SOURCE TERMS KERNEL
# ==============================================================================

@jit(nopython=True, cache=True, fastmath=True, parallel=True)
def compute_source_terms_kernel(
    N, alpha, beta_U, e6phi, gamma_LL,
    TUU_00, TUU_0i, TUU_ij, TUD_0j,
    K_LL, dalpha_dx, hatD_beta_U, hatD_gamma_LL,
    src_S_out, src_tau_out
):
    """
    Compute GRHD source terms with explicit loops (replaces einsum).

    Energy source:
        S_τ = α e^{6φ} [K_ij (T^{00} β^i β^j + 2 T^{0i} β^j + T^{ij})
                       - (T^{00} β^i + T^{0i}) ∂_i α]

    Momentum source:
        S_{S_i} = α e^{6φ} [-T^{00} α ∂_i α
                          + T^0_j (∂_i β^j + Γ̂^j_{ik} β^k)  <- hatD_beta_U
                          + (1/2) (T^{00} β^j β^k + 2 T^{0j} β^k + T^{jk}) ∇̂_i γ_{jk}]

    Args:
        N: Number of grid points
        alpha: (N,) lapse
        beta_U: (N, 3) shift vector
        e6phi: (N,) e^{6φ}
        gamma_LL: (N, 3, 3) spatial metric
        TUU_00: (N,) T^{00}
        TUU_0i: (N, 3) T^{0i}
        TUU_ij: (N, 3, 3) T^{ij}
        TUD_0j: (N, 3) T^0_j
        K_LL: (N, 3, 3) extrinsic curvature
        dalpha_dx: (N, 3) lapse derivatives
        hatD_beta_U: (N, 3, 3) covariant derivative of shift ∇̂_i β^j
        hatD_gamma_LL: (N, 3, 3, 3) covariant derivative of metric ∇̂_i γ_{jk}
        src_S_out: (N, 3) output momentum source
        src_tau_out: (N,) output energy source
    """
    for m in prange(N):
        alph = alpha[m]
        e6p = e6phi[m]
        T00 = TUU_00[m]

        # ================================================================
        # ENERGY SOURCE TERM
        # ================================================================

        # Term 1: K_ij * tensor_block_ij
        # tensor_block_ij = T^{00} β^i β^j + 2 T^{0i} β^j + T^{ij}
        term1_tau = 0.0
        for i in range(SPACEDIM):
            for j in range(SPACEDIM):
                tensor_ij = (T00 * beta_U[m, i] * beta_U[m, j]
                            + 2.0 * TUU_0i[m, i] * beta_U[m, j]
                            + TUU_ij[m, i, j])
                term1_tau += K_LL[m, i, j] * tensor_ij

        # Term 2: -(T^{00} β^i + T^{0i}) ∂_i α
        term2_tau = 0.0
        for i in range(SPACEDIM):
            term2_tau -= (T00 * beta_U[m, i] + TUU_0i[m, i]) * dalpha_dx[m, i]

        src_tau_out[m] = alph * e6p * (term1_tau + term2_tau)

        # ================================================================
        # MOMENTUM SOURCE TERM
        # ================================================================
        for i in range(SPACEDIM):
            # Term 1: -T^{00} α ∂_i α
            first_term = -T00 * alph * dalpha_dx[m, i]

            # Term 2: T^0_j ∇̂_i β^j
            second_term = 0.0
            for j in range(SPACEDIM):
                second_term += TUD_0j[m, j] * hatD_beta_U[m, i, j]

            # Term 3: (1/2) tensor_block_jk * ∇̂_i γ_{jk}
            # tensor_block_jk = T^{00} β^j β^k + 2 T^{0j} β^k + T^{jk}
            third_term = 0.0
            for j in range(SPACEDIM):
                for k in range(SPACEDIM):
                    tensor_jk = (T00 * beta_U[m, j] * beta_U[m, k]
                                + 2.0 * TUU_0i[m, j] * beta_U[m, k]
                                + TUU_ij[m, j, k])
                    third_term += tensor_jk * hatD_gamma_LL[m, i, j, k]
            third_term *= 0.5

            src_S_out[m, i] = alph * e6p * (first_term + second_term + third_term)


# ==============================================================================
# CONNECTION TERMS KERNEL
# ==============================================================================

@jit(nopython=True, cache=True, fastmath=True, parallel=True)
def compute_connection_terms_kernel(
    N, hat_chris, fD_U, fTau_U, fS_UD,
    conn_D_out, conn_S_out, conn_tau_out
):
    """
    Compute connection term contributions from reference metric Christoffel symbols.

    Connection terms:
        D, τ:   -Γ̂^k_{kj} F̃^j
        S_i:    -Γ̂^k_{kj} F̃^j_i + Γ̂^l_{ji} F̃^j_l

    Args:
        N: Number of grid points
        hat_chris: (N, 3, 3, 3) reference metric Christoffel symbols Γ̂^i_{jk}
        fD_U: (N, 3) density partial flux vector
        fTau_U: (N, 3) energy partial flux vector
        fS_UD: (N, 3, 3) momentum partial flux tensor F̃^j_i
        conn_D_out: (N,) output connection term for D
        conn_S_out: (N, 3) output connection term for S_i
        conn_tau_out: (N,) output connection term for tau
    """
    for m in prange(N):
        # Compute trace: Γ̂^k_{kj} (sum over first index k)
        Gamma_trace = np.empty(SPACEDIM)
        for j in range(SPACEDIM):
            Gamma_trace[j] = 0.0
            for k in range(SPACEDIM):
                Gamma_trace[j] += hat_chris[m, k, k, j]

        # D equation: -Γ̂^k_{kj} F̃^j_D
        conn_D_out[m] = 0.0
        for j in range(SPACEDIM):
            conn_D_out[m] -= Gamma_trace[j] * fD_U[m, j]

        # τ equation: -Γ̂^k_{kj} F̃^j_τ
        conn_tau_out[m] = 0.0
        for j in range(SPACEDIM):
            conn_tau_out[m] -= Gamma_trace[j] * fTau_U[m, j]

        # S_i equation: -Γ̂^k_{kj} F̃^j_i + Γ̂^l_{ji} F̃^j_l
        for i in range(SPACEDIM):
            # First term: -Γ̂^k_{kj} F̃^j_i = -Gamma_trace[j] * fS_UD[j, i]
            term1 = 0.0
            for j in range(SPACEDIM):
                term1 -= Gamma_trace[j] * fS_UD[m, j, i]

            # Second term: Γ̂^l_{ji} F̃^j_l
            term2 = 0.0
            for j in range(SPACEDIM):
                for l in range(SPACEDIM):
                    term2 += hat_chris[m, l, j, i] * fS_UD[m, j, l]

            conn_S_out[m, i] = term1 + term2


# ==============================================================================
# FLUX COMPUTATION KERNEL
# ==============================================================================

@jit(nopython=True, cache=True, fastmath=True, parallel=True)
def compute_fluxes_kernel(
    N, rho0, v_U, pressure, W, h,
    alpha, e6phi, gamma_LL, gamma_UU, beta_U,
    fD_U_out, fTau_U_out, fS_UD_out
):
    """
    Compute densitized partial flux vectors for Valencia formulation.

    Flux definitions:
        F̃D^j    = e^{6φ} α ρ₀ W ṽ^j
        F̃_S^j_i = e^{6φ} α T^j_i
        F̃τ^j    = e^{6φ} (α² T^{0j} - α ρ₀ W ṽ^j)

    where ṽ^j = v^j - β^j/α

    Args:
        N: Number of grid points
        rho0, pressure, W, h: (N,) primitive quantities
        v_U: (N, 3) spatial velocity
        alpha: (N,) lapse
        e6phi: (N,) e^{6φ}
        gamma_LL: (N, 3, 3) spatial metric
        gamma_UU: (N, 3, 3) inverse spatial metric
        beta_U: (N, 3) shift vector
        fD_U_out: (N, 3) output density flux
        fTau_U_out: (N, 3) output energy flux
        fS_UD_out: (N, 3, 3) output momentum flux tensor
    """
    for m in prange(N):
        alph = alpha[m]
        alph_sq = alph * alph
        e6p = e6phi[m]
        rho = rho0[m]
        Wm = W[m]
        hm = h[m]
        pm = pressure[m]
        rho_h = rho * hm
        D = rho * Wm  # Conservative density

        # Valencia velocity: ṽ^i = v^i - β^i/α
        vtilde = np.empty(SPACEDIM)
        for i in range(SPACEDIM):
            vtilde[i] = v_U[m, i] - beta_U[m, i] / alph

        # 4-velocity: u^0 = W/α, u^i = W ṽ^i
        u0 = Wm / alph
        ui = np.empty(SPACEDIM)
        for i in range(SPACEDIM):
            ui[i] = Wm * vtilde[i]

        # Inverse 4-metric spatial part: g^{ij} = γ^{ij} - β^i β^j/α²
        g4UU_ij = np.empty((SPACEDIM, SPACEDIM))
        for i in range(SPACEDIM):
            for j in range(SPACEDIM):
                g4UU_ij[i, j] = gamma_UU[m, i, j] - beta_U[m, i] * beta_U[m, j] / alph_sq

        # g^{0i} = β^i/α²
        g4UU_0i = np.empty(SPACEDIM)
        for i in range(SPACEDIM):
            g4UU_0i[i] = beta_U[m, i] / alph_sq

        # T^{0i} = ρ₀ h u^0 u^i + P g^{0i}
        TUU_0i = np.empty(SPACEDIM)
        for i in range(SPACEDIM):
            TUU_0i[i] = rho_h * u0 * ui[i] + pm * g4UU_0i[i]

        # v_i = γ_{ij} v^j (covariant velocity components)
        vi = np.empty(SPACEDIM)
        for i in range(SPACEDIM):
            vi[i] = 0.0
            for j in range(SPACEDIM):
                vi[i] += gamma_LL[m, i, j] * v_U[m, j]

        # Density flux: F̃^j_D = e^{6φ} α D ṽ^j
        for j in range(SPACEDIM):
            fD_U_out[m, j] = e6p * alph * D * vtilde[j]

        # Energy flux: F̃^j_τ = e^{6φ} (α² T^{0j} - α D ṽ^j)
        for j in range(SPACEDIM):
            fTau_U_out[m, j] = e6p * (alph_sq * TUU_0i[j] - alph * D * vtilde[j])

        # Momentum flux: F̃^j_i = e^{6φ} α T^j_i
        # T^j_i using equation (27): W² ρh v_i ṽ^j + p δ^j_i
        for j in range(SPACEDIM):
            for i in range(SPACEDIM):
                TUD_ji = rho_h * Wm * Wm * vi[i] * vtilde[j]
                if i == j:
                    TUD_ji += pm  # + p δ^j_i
                fS_UD_out[m, j, i] = e6p * alph * TUD_ji


# ==============================================================================
# COVARIANT DERIVATIVE OF SHIFT KERNEL
# ==============================================================================

@jit(nopython=True, cache=True, fastmath=True, parallel=True)
def compute_hatD_beta_kernel(N, dbeta_dx, hat_chris, beta_U, hatD_beta_out):
    """
    Compute covariant derivative of shift: ∇̂_i β^j = ∂_i β^j + Γ̂^j_{ik} β^k

    Args:
        N: Number of grid points
        dbeta_dx: (N, 3, 3) partial derivatives ∂_i β^j (note: transposed input)
        hat_chris: (N, 3, 3, 3) reference metric Christoffel symbols Γ̂^i_{jk}
        beta_U: (N, 3) shift vector
        hatD_beta_out: (N, 3, 3) output ∇̂_i β^j
    """
    for m in prange(N):
        for i in range(SPACEDIM):
            for j in range(SPACEDIM):
                # ∂_i β^j (from transposed input)
                hatD_beta_out[m, i, j] = dbeta_dx[m, i, j]

                # + Γ̂^j_{ik} β^k
                for k in range(SPACEDIM):
                    hatD_beta_out[m, i, j] += hat_chris[m, j, i, k] * beta_U[m, k]


# ==============================================================================
# COVARIANT DERIVATIVE OF METRIC KERNEL
# ==============================================================================

@jit(nopython=True, cache=True, fastmath=True, parallel=True)
def compute_hatD_gamma_kernel(N, e4phi, dphi_dx, bar_gamma_LL, hat_D_bar_gamma,
                               hatD_gamma_out):
    """
    Compute covariant derivative of metric: ∇̂_i γ_{jk} = e^{4φ} [4 γ̄_{jk} ∂_i φ + ∇̂_i γ̄_{jk}]

    Args:
        N: Number of grid points
        e4phi: (N,) e^{4φ}
        dphi_dx: (N, 3) derivatives of conformal factor
        bar_gamma_LL: (N, 3, 3) conformal metric
        hat_D_bar_gamma: (N, 3, 3, 3) covariant derivative of conformal metric (j, k, i)
        hatD_gamma_out: (N, 3, 3, 3) output ∇̂_i γ_{jk} with shape (N, i, j, k)
    """
    for m in prange(N):
        e4p = e4phi[m]
        for i in range(SPACEDIM):
            for j in range(SPACEDIM):
                for k in range(SPACEDIM):
                    # ∇̂_i γ_{jk} = e^{4φ} [4 γ̄_{jk} ∂_i φ + ∇̂_i γ̄_{jk}]
                    # hat_D_bar_gamma has shape (N, j, k, i), need to transpose
                    hatD_gamma_out[m, i, j, k] = e4p * (
                        4.0 * bar_gamma_LL[m, j, k] * dphi_dx[m, i]
                        + hat_D_bar_gamma[m, j, k, i]
                    )


# ==============================================================================
# FULLY FUSED SOURCE TERMS KERNEL
# ==============================================================================

@jit(nopython=True, cache=True, fastmath=True, parallel=True)
def compute_source_terms_fused_kernel(
    # Primitive variables
    rho0, v_U, pressure, W, h,
    # Geometry (already computed)
    alpha, beta_U, e4phi, e6phi, gamma_LL, gamma_UU,
    # BSSN variables
    K_trace, bar_A_LL,
    # Derivatives
    dalpha_dx, dphi_dx, dbeta_dx_transposed, hat_D_bar_gamma,
    # Background
    hat_chris,
    # Output
    src_S_out, src_tau_out
):
    """
    Fully fused source terms computation using only scalar temporaries.

    Avoids all array allocations inside the loop for maximum performance.
    All intermediate values are computed on-the-fly or stored in scalars.
    """
    N = rho0.shape[0]

    for m in prange(N):
        # ================================================================
        # LOCAL SCALAR VARIABLES
        # ================================================================
        alph = alpha[m]
        alph_sq = alph * alph
        inv_alph_sq = 1.0 / alph_sq
        e4p = e4phi[m]
        e6p = e6phi[m]
        inv_e4p = 1.0 / e4p
        rho_h = rho0[m] * h[m]
        pm = pressure[m]
        Wm = W[m]
        K_third = K_trace[m] / 3.0

        # Load beta components
        beta0 = beta_U[m, 0]
        beta1 = beta_U[m, 1]
        beta2 = beta_U[m, 2]

        # Load velocity components
        v0 = v_U[m, 0]
        v1 = v_U[m, 1]
        v2 = v_U[m, 2]

        # ================================================================
        # 1. COMPUTE 4-VELOCITY (scalar components)
        # ================================================================
        u0 = Wm / alph
        inv_alph = 1.0 / alph
        u1 = Wm * (v0 - beta0 * inv_alph)
        u2 = Wm * (v1 - beta1 * inv_alph)
        u3 = Wm * (v2 - beta2 * inv_alph)

        # ================================================================
        # 2. INVERSE 4-METRIC COMPONENTS (computed on-the-fly)
        # ================================================================
        g4UU_00 = -inv_alph_sq
        g4UU_01 = beta0 * inv_alph_sq
        g4UU_02 = beta1 * inv_alph_sq
        g4UU_03 = beta2 * inv_alph_sq

        # ================================================================
        # 3. COMPUTE STRESS-ENERGY TENSOR T^{μν}
        # ================================================================
        T00 = rho_h * u0 * u0 + pm * g4UU_00

        # T^{0i}
        T01 = rho_h * u0 * u1 + pm * g4UU_01
        T02 = rho_h * u0 * u2 + pm * g4UU_02
        T03 = rho_h * u0 * u3 + pm * g4UU_03

        # T^{ij} computed on demand below

        # ================================================================
        # 4. COMPUTE g_{0j} = β_j = γ_{jk} β^k
        # ================================================================
        g4DD_01 = gamma_LL[m, 0, 0] * beta0 + gamma_LL[m, 0, 1] * beta1 + gamma_LL[m, 0, 2] * beta2
        g4DD_02 = gamma_LL[m, 1, 0] * beta0 + gamma_LL[m, 1, 1] * beta1 + gamma_LL[m, 1, 2] * beta2
        g4DD_03 = gamma_LL[m, 2, 0] * beta0 + gamma_LL[m, 2, 1] * beta1 + gamma_LL[m, 2, 2] * beta2

        # ================================================================
        # 5. COMPUTE T^0_j = T^{00} g_{0j} + T^{0i} g_{ij}
        # ================================================================
        T0_1 = T00 * g4DD_01 + T01 * gamma_LL[m, 0, 0] + T02 * gamma_LL[m, 1, 0] + T03 * gamma_LL[m, 2, 0]
        T0_2 = T00 * g4DD_02 + T01 * gamma_LL[m, 0, 1] + T02 * gamma_LL[m, 1, 1] + T03 * gamma_LL[m, 2, 1]
        T0_3 = T00 * g4DD_03 + T01 * gamma_LL[m, 0, 2] + T02 * gamma_LL[m, 1, 2] + T03 * gamma_LL[m, 2, 2]

        # ================================================================
        # 6. COMPUTE ENERGY SOURCE TERM S_τ
        # ================================================================
        term1_tau = 0.0
        term2_tau = 0.0

        for i in range(SPACEDIM):
            beta_i = beta_U[m, i]
            T0i_val = T01 if i == 0 else (T02 if i == 1 else T03)
            ui_val = u1 if i == 0 else (u2 if i == 1 else u3)

            for j in range(SPACEDIM):
                beta_j = beta_U[m, j]
                T0j_val = T01 if j == 0 else (T02 if j == 1 else T03)
                uj_val = u1 if j == 0 else (u2 if j == 1 else u3)

                # T^{ij} = ρh u^i u^j + P g^{ij}
                g4UU_ij = gamma_UU[m, i, j] - beta_i * beta_j * inv_alph_sq
                Tij_val = rho_h * ui_val * uj_val + pm * g4UU_ij

                # K_{ij} = e^{4φ} Ā_{ij} + (K/3) γ_{ij}
                K_ij = e4p * bar_A_LL[m, i, j] + K_third * gamma_LL[m, i, j]

                # tensor_block_ij = T^{00} β^i β^j + 2 T^{0i} β^j + T^{ij}
                tensor_ij = T00 * beta_i * beta_j + 2.0 * T0i_val * beta_j + Tij_val
                term1_tau += K_ij * tensor_ij

            # Term 2: -(T^{00} β^i + T^{0i}) ∂_i α
            term2_tau -= (T00 * beta_i + T0i_val) * dalpha_dx[m, i]

        src_tau_out[m] = alph * e6p * (term1_tau + term2_tau)

        # ================================================================
        # 7. COMPUTE MOMENTUM SOURCE TERM S_i
        # ================================================================
        for i in range(SPACEDIM):
            # Term 1: -T^{00} α ∂_i α
            first_term = -T00 * alph * dalpha_dx[m, i]

            # Term 2: T^0_j ∇̂_i β^j
            second_term = 0.0
            T0_vals = (T0_1, T0_2, T0_3)
            for j in range(SPACEDIM):
                # ∇̂_i β^j = ∂_i β^j + Γ̂^j_{ik} β^k
                hatD_beta_ij = dbeta_dx_transposed[m, i, j]
                for k in range(SPACEDIM):
                    hatD_beta_ij += hat_chris[m, j, i, k] * beta_U[m, k]
                second_term += T0_vals[j] * hatD_beta_ij

            # Term 3: (1/2) tensor_block_jk * ∇̂_i γ_{jk}
            third_term = 0.0
            dphi_i = dphi_dx[m, i]
            for j in range(SPACEDIM):
                beta_j = beta_U[m, j]
                T0j_val = T01 if j == 0 else (T02 if j == 1 else T03)
                uj_val = u1 if j == 0 else (u2 if j == 1 else u3)

                for k in range(SPACEDIM):
                    beta_k = beta_U[m, k]
                    T0k_val = T01 if k == 0 else (T02 if k == 1 else T03)
                    uk_val = u1 if k == 0 else (u2 if k == 1 else u3)

                    # T^{jk}
                    g4UU_jk = gamma_UU[m, j, k] - beta_j * beta_k * inv_alph_sq
                    Tjk_val = rho_h * uj_val * uk_val + pm * g4UU_jk

                    # tensor_block_jk
                    tensor_jk = T00 * beta_j * beta_k + 2.0 * T0j_val * beta_k + Tjk_val

                    # ∇̂_i γ_{jk} = e^{4φ} [4 γ̄_{jk} ∂_i φ + ∇̂_i γ̄_{jk}]
                    bar_gamma_jk = gamma_LL[m, j, k] * inv_e4p
                    hatD_gamma_ijk = e4p * (4.0 * bar_gamma_jk * dphi_i + hat_D_bar_gamma[m, j, k, i])

                    third_term += tensor_jk * hatD_gamma_ijk

            third_term *= 0.5

            src_S_out[m, i] = alph * e6p * (first_term + second_term + third_term)

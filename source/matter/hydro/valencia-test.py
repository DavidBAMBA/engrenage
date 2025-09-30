
# valencia_reference_metric.py
"""
Valencia formulation with reference metric for relativistic hydrodynamics.

Implements the full reference-metric approach following the BSSN style:
- Uses hat Christoffel symbols from background
- Computes connection terms via tensor contractions (einsum)
- Source terms use covariant derivatives and stress-energy tensor
- Spherical symmetry is implicit through background definitions

Conservation equations (general form in reference-metric form):
    ∂_t(J D)   + D̂_j[(f_D)^j]   = 0
    ∂_t(J S_i) + D̂_j[(f_S)^j_i] = (s_S)_i
    ∂_t(J τ)   + D̂_j[(f_τ)^j]   = s_τ

Implementation with partial derivatives:
    D̂_j V^j               = (1/√ĝ) ∂_j[ √ĝ V^j ]
    D̂_j T^{j}{}_i         = (1/√ĝ) ∂_j[ √ĝ T^{j}{}_i ] - Γ̂^{k}{}_{ji} T^{j}{}_k

=> En el código:
- Masa y energía: la divergencia densitizada (1/√ĝ)∂[√ĝ··] YA incluye la traza Γ̂^j_{jk} V^k.
  No añadir términos extra.
- Momento: añadir SOLO + Γ̂^{k}{}_{ji} (f_S)^{j}{}_k.

Minkowski (α=1, β^i=0, K_{ij}=0):
- Fuentes de espaciotiempo s_τ = 0, (s_S)_i = 0.
- El término 2p/r en 1D esféricas surge del término +Γ̂^{k}{}_{jr}(f_S)^{j}{}_k, no de s_S.
"""

import numpy as np

from source.bssn.tensoralgebra import (
    SPACEDIM,
    get_bar_gamma_LL,
    get_det_bar_gamma,
    get_bar_A_LL,
    get_hat_D_bar_gamma_LL
)
from source.backgrounds.sphericalbackground import i_r, i_t, i_p
from source.core.spacing import NUM_GHOSTS
from source.matter.hydro.cons2prim import prim_to_cons


class ValenciaReferenceMetric:
    """
    Valencia formulation with reference metric.

    Uses Engrenage-style divergence operators with metric densitization:
        div_hat(F) ≡ -(1/√ĝ) ∂_r [ √ĝ · (α J · F_phys) ]
    and adds ONLY the missing connection piece for the momentum equation.
    """

    # ============================================================================
    # MAIN RHS COMPUTATION
    # ============================================================================

    def compute_rhs(self, D, Sr, tau, rho0, vr, pressure, W, h,
                    r, bssn_vars, bssn_d1, background, spacetime_mode,
                    eos, grid, reconstructor, riemann_solver):
        """
        Compute RHS for Valencia equations in reference metric formulation.

        Args:
            D, Sr, tau: Conservative variables
            rho0, vr, pressure, W, h: Primitive variables
            r: Radial coordinate array
            bssn_vars: BSSN state variables
            bssn_d1: First derivatives of BSSN variables
            background: Background metric (spherical)
            spacetime_mode: "fixed_minkowski" or "dynamic"
            eos: Equation of state
            grid: Grid object with derivative operators
            reconstructor: Reconstruction method
            riemann_solver: Riemann solver

        Returns:
            (dD/dt, dSr/dt, dτ/dt) - RHS arrays
        """
        # Mesh spacing
        dr = self._get_mesh_spacing(grid, r)
        N = len(r)

        # Apply boundary conditions to conservative variables
        D, Sr, tau = D.copy(), Sr.copy(), tau.copy()
        D, Sr, tau = self._apply_ghost_cell_boundaries(D, Sr, tau, r)

        # Extract geometry from BSSN variables
        g = self._extract_geometry(r, bssn_vars, spacetime_mode, background)

        # Compute densitized fluxes at cell faces: √ĝ (α J) F_phys
        flux_hat = self._compute_interface_fluxes(
            rho0, vr, pressure, g, r, eos, reconstructor, riemann_solver
        )

        # ======================================================================
        # PART 1: Divergence terms (Engrenage operator)
        #   -(1/√ĝ) ∂_r [ √ĝ (αJ F_phys) ]
        # ======================================================================
        div_D = self._compute_flux_divergence(
            flux_hat['D'], g['sqrt_g_hat_cell'], dr, grid
        )
        div_Sr = self._compute_flux_divergence(
            flux_hat['Sr'], g['sqrt_g_hat_cell'], dr, grid
        )
        div_tau = self._compute_flux_divergence(
            flux_hat['tau'], g['sqrt_g_hat_cell'], dr, grid
        )

        # ======================================================================
        # PART 2: Connection correction terms
        #   Only needed for momentum (mixed tensor), with +Γ̂^k_{ji} f^j{}_k
        # ======================================================================
        conn_D, conn_Sr, conn_tau = self._compute_connection_terms(
            D, Sr, tau, rho0, vr, pressure, W, h, g, background
        )

        # ======================================================================
        # PART 3: Physical source terms (spacetime geometry)
        # ======================================================================
        src_Sr, src_tau = self._compute_source_terms(
            rho0, vr, pressure, W, h, g, bssn_vars, bssn_d1,
            background, spacetime_mode, r
        )

        # ======================================================================
        # ASSEMBLE FULL RHS
        #   RHS_U = div_hat + conn + src; then divide by J to get ∂_t U
        # ======================================================================
        rhs_D = np.zeros(N)
        rhs_Sr = np.zeros(N)
        rhs_tau = np.zeros(N)

        # Interior points: divergence + connections + sources
        rhs_D[1:-1] = div_D + conn_D[1:-1]
        rhs_Sr[1:-1] = div_Sr + conn_Sr[1:-1] + src_Sr[1:-1]
        rhs_tau[1:-1] = div_tau + conn_tau[1:-1] + src_tau[1:-1]

        # Divide by Jacobian J to get ∂_t(conserved variables)
        J = g['J_cell'] + 1e-30
        rhs_D /= J
        rhs_Sr /= J
        rhs_tau /= J

        # Apply boundary conditions to RHS
        rhs_D, rhs_Sr, rhs_tau = self._apply_rhs_boundary_conditions(
            rhs_D, rhs_Sr, rhs_tau, r
        )

        return rhs_D, rhs_Sr, rhs_tau

    # ============================================================================
    # CONNECTION TERMS
    # ============================================================================

    def _compute_connection_terms(self, D, Sr, tau, rho0, vr, pressure, W, h,
                                  g, background):
        """
        Compute connection correction terms using hat Christoffel symbols.

        Using the Engrenage divergence with √ĝ-densitization:
          D̂_j V^j         = (1/√ĝ) ∂_j(√ĝ V^j)         [already in div term]
          D̂_j T^{j}{}_i   = (1/√ĝ) ∂_j(√ĝ T^{j}{}_i) - Γ̂^{k}{}_{ji} T^{j}{}_k

        Therefore:
          - Mass, Energy: NO extra connection terms (already included).
          - Momentum: add ONLY + Γ̂^{k}{}_{ji} (f_S)^{j}{}_k.

        Returns:
            (conn_D, conn_Sr, conn_tau) arrays of length N.
        """
        N = len(D)
        alpha, beta_r = g['alpha'], g['beta_r']
        J_cell = g['J_cell']

        # Build coordinate velocity v^i = (v^r, 0, 0)
        v_coord = np.zeros((N, SPACEDIM))
        v_coord[:, i_r] = vr

        # Valencia velocity: v̄^i = v^i - β^i/α
        beta_U = np.zeros((N, SPACEDIM))
        beta_U[:, i_r] = beta_r
        v_tilde = v_coord.copy()
        v_tilde[:, i_r] = v_coord[:, i_r] - beta_U[:, i_r] / (alpha + 1e-30)

        # Mixed momentum flux (without √ĝ): (f_S)^j{}_i = α J [ S_i v̄^j + p δ^j_i ]
        gamma_rr = g['gamma_rr']
        v_lower = np.zeros((N, SPACEDIM))
        v_lower[:, i_r] = gamma_rr * vr  # v_r = γ_rr v^r

        S_lower = np.zeros((N, SPACEDIM))
        S_lower[:, i_r] = rho0 * h * W * W * v_lower[:, i_r]

        fS_mixed = np.zeros((N, SPACEDIM, SPACEDIM))  # indices (j, i)
        for j in range(SPACEDIM):
            for i in range(SPACEDIM):
                fS_mixed[:, j, i] = (alpha * J_cell) * S_lower[:, i] * v_tilde[:, j]
                if i == j:
                    fS_mixed[:, j, i] += (alpha * J_cell) * pressure

        # Christoffel symbols: Γ̂^i_{jk} with shape (N, i, j, k)
        hat_chris = background.hat_christoffel

        # --- Mass & Energy: no correction (already included in densitized div)
        conn_D = np.zeros_like(D)
        conn_tau = np.zeros_like(tau)

        # --- Momentum: + Γ̂^{k}{}_{ji} (f_S)^{j}{}_k
        conn_S_vector = np.einsum('xjk,xkji->xi', fS_mixed, hat_chris)
        conn_Sr = conn_S_vector[:, i_r]

        return conn_D, conn_Sr, conn_tau

    # ============================================================================
    # SOURCE TERMS
    # ============================================================================

    def _compute_source_terms(self, rho0, vr, pressure, W, h, g,
                              bssn_vars, bssn_d1, background, spacetime_mode, r):
        """
        Compute physical source terms from spacetime geometry.

        Momentum source:
            (s_S)_i = α J [-T^{00} α ∂_i α + T^{0}_k D̂_i β^k
                           + (1/2) Σ_{jk}(T^{00}β^j β^k + 2T^{0j}β^k + T^{jk}) D̂_i γ_{jk}]

        Energy source:
            s_τ = α J [T^{00}(β^i β^j K_{ij} - β^i ∂_i α)
                       + T^{0i}(2β^j K_{ij} - ∂_i α) + T^{ij}K_{ij}]

        For fixed Minkowski (α=1, β^i=0, K_{ij}=0): s_τ = 0, (s_S)_i = 0.
        """
        if spacetime_mode == "fixed_minkowski":
            return self._compute_source_terms_minkowski(
                rho0, vr, pressure, W, h, g, r, background
            )
        else:
            return self._compute_source_terms_curved(
                rho0, vr, pressure, W, h, g, bssn_vars, bssn_d1,
                background, r
            )

    def _compute_source_terms_minkowski(self, rho0, vr, pressure, W, h,
                                        g, r, background):
        """
        Source terms for fixed Minkowski spacetime.
        α=1, β^i=0, K_{ij}=0 => s_τ = 0 and (s_S)_i = 0.
        """
        N = len(r)
        src_tau = np.zeros(N)
        src_Sr = np.zeros(N)
        return src_Sr, src_tau

    def _compute_source_terms_curved(self, rho0, vr, pressure, W, h, g,
                                     bssn_vars, bssn_d1, background, r):
        """
        Source terms for curved spacetime (general case).
        Uses full stress-energy tensor and spacetime derivatives.
        """
        N = len(r)
        alpha, beta_r = g['alpha'], g['beta_r']
        gamma_rr = g['gamma_rr']
        J = g['J_cell']

        # --- Stress-energy tensor components (perfect fluid)
        T00, T0r, Trr = self._compute_stress_energy_tensor(
            rho0, vr, pressure, W, h, alpha, beta_r, gamma_rr
        )

        # --- Spatial metric γ_{ij} from BSSN
        phi = np.asarray(bssn_vars.phi, dtype=float)
        e4phi = np.exp(4.0 * phi)
        bar_gamma_LL = get_bar_gamma_LL(r, bssn_vars.h_LL, background)
        gamma_LL = e4phi[:, None, None] * bar_gamma_LL
        gamma_UU = np.linalg.inv(gamma_LL)

        # --- Extrinsic curvature K_{ij}
        K = np.asarray(bssn_vars.K, dtype=float)
        bar_A_LL = get_bar_A_LL(r, bssn_vars, background)
        K_LL = e4phi[:, None, None] * bar_A_LL + (K / 3.0)[:, None, None] * gamma_LL

        # --- Metric derivatives
        dalpha_dx = np.zeros((N, SPACEDIM))
        if hasattr(bssn_d1, 'lapse') and bssn_d1.lapse is not None:
            dalpha_der = np.asarray(bssn_d1.lapse)
            if dalpha_der.ndim >= 2:
                dalpha_dx[:, i_r] = dalpha_der[:, i_r]

        # D̂_r β^j
        Dhat_beta_U = self._compute_Dhat_beta(bssn_vars, bssn_d1, background, r)

        # D̂_r γ_{jk}
        Dhat_gamma_LL = self._compute_Dhat_gamma(
            bssn_vars, bssn_d1, background, r, phi, bar_gamma_LL
        )

        # --- Build tensor quantities
        beta_U = np.zeros((N, SPACEDIM))
        beta_U[:, i_r] = beta_r

        beta_lower = np.einsum('xij,xj->xi', gamma_LL, beta_U)  # β_i = γ_{ij} β^j
        T0_lower = np.zeros((N, SPACEDIM))                      # T^{0}{}_i
        T0_lower[:, i_r] = T00 * beta_lower[:, i_r] + T0r * gamma_LL[:, i_r, i_r]

        T0U = np.zeros((N, SPACEDIM))  # T^{0i}
        T0U[:, i_r] = T0r

        TUU = np.zeros((N, SPACEDIM, SPACEDIM))  # T^{ij}
        TUU[:, i_r, i_r] = Trr
        TUU[:, i_t, i_t] = pressure * gamma_UU[:, i_t, i_t]  # T^{θθ} = p g^{θθ}
        TUU[:, i_p, i_p] = pressure * gamma_UU[:, i_p, i_p]  # T^{φφ} = p g^{φφ}

        tensor_block = (
            T00[:, None, None] * np.einsum('xi,xj->xij', beta_U, beta_U)
            + 2.0 * np.einsum('xi,xj->xij', T0U, beta_U)
            + TUU
        )

        # --- Momentum source (s_S)_r
        src_Sr_vector = alpha * J * (
            -T00 * alpha * dalpha_dx[:, i_r]                                  # -T^{00} α ∂_r α
            + np.einsum('xi,xi->x', T0_lower, Dhat_beta_U)                    # T^{0}{}_k D̂_r β^k
            + 0.5 * np.einsum('xjk,xjk->x', tensor_block, Dhat_gamma_LL)      # ½ Σ T^{jk} D̂_r γ_{jk}
        )
        src_Sr = src_Sr_vector

        # --- Energy source s_τ
        src_tau = alpha * J * (
            T00 * np.einsum('xi,xj,xij->x', beta_U, beta_U, K_LL)             # T^{00} β^i β^j K_{ij}
            - T00 * np.einsum('xi,xi->x', beta_U, dalpha_dx)                  # -T^{00} β^i ∂_i α
            + 2.0 * np.einsum('xi,xj,xij->x', T0U, beta_U, K_LL)              # 2T^{0i} β^j K_{ij}
            - np.einsum('xi,xi->x', T0U, dalpha_dx)                           # -T^{0i} ∂_i α
            + np.einsum('xij,xij->x', TUU, K_LL)                              # T^{ij} K_{ij}
        )

        return src_Sr, src_tau

    # ============================================================================
    # GEOMETRY EXTRACTION
    # ============================================================================

    def _extract_geometry(self, r, bssn_vars, spacetime_mode, background):
        """
        Extract geometric quantities from BSSN variables.

        Returns dict with:
            alpha, beta_r: Lapse and radial shift
            gamma_rr: Radial metric component
            e6phi: Conformal factor exp(6φ)
            J_cell, J_face: Jacobian at cells and faces
            sqrt_g_hat_cell, sqrt_g_hat_face: √det(ĝ) at cells and faces
        """
        N = len(r)
        g = {}

        if spacetime_mode == "fixed_minkowski":
            g['alpha'] = np.ones(N)
            g['beta_r'] = np.zeros(N)
            g['e6phi'] = np.ones(N)
            g['gamma_rr'] = np.ones(N)
        else:
            # Lapse α
            g['alpha'] = np.asarray(bssn_vars.lapse, dtype=float)

            # Shift β^r (contravariant)
            if hasattr(bssn_vars, 'shift_U') and bssn_vars.shift_U is not None:
                shift_array = np.asarray(bssn_vars.shift_U)
                g['beta_r'] = (shift_array[:, i_r].astype(float)
                               if shift_array.ndim >= 2 else np.zeros(N))
            else:
                g['beta_r'] = np.zeros(N)

            # Conformal factor
            phi_arr = np.asarray(bssn_vars.phi, dtype=float)
            g['e6phi'] = np.exp(6.0 * phi_arr)

            # Spatial metric γ_{rr} = e^{4φ} γ̄_{rr}
            bar_gamma_LL = get_bar_gamma_LL(r, bssn_vars.h_LL, background)
            e4phi = np.exp(4.0 * phi_arr)
            g['gamma_rr'] = e4phi * bar_gamma_LL[:, i_r, i_r]

        # √det(ĝ) for flat spherical: √det(ĝ) = r²
        r_arr = np.asarray(r, dtype=float)
        r_face = 0.5 * (r_arr[:-1] + r_arr[1:])
        g['sqrt_g_hat_cell'] = np.maximum(np.abs(r_arr), 1e-30) ** 2
        g['sqrt_g_hat_face'] = np.maximum(np.abs(r_face), 1e-30) ** 2

        # Jacobian J = e^{6φ} √(γ̄/ĝ)
        if spacetime_mode != "fixed_minkowski":
            det_bar_gamma = get_det_bar_gamma(r, bssn_vars.h_LL, background)
            sqrt_bar_gamma = np.sqrt(np.abs(det_bar_gamma) + 1e-30)
            sqrt_hat_gamma = np.sqrt(np.abs(background.det_hat_gamma) + 1e-30)
            J_cell = g['e6phi'] * sqrt_bar_gamma / sqrt_hat_gamma
        else:
            J_cell = np.ones(N)

        g['J_cell'] = J_cell
        g['J_face'] = 0.5 * (J_cell[:-1] + J_cell[1:])

        return g

    # ============================================================================
    # FLUX COMPUTATION
    # ============================================================================

    def _compute_interface_fluxes(self, rho0, vr, pressure, g, r,
                                  eos, reconstructor, riemann_solver):
        """
        Compute densitized fluxes at cell faces: √ĝ (α J) F_phys.

        The physical fluxes F_phys are computed by the Riemann solver.
        We then multiply by α J (spacetime factors) and √ĝ (background metric).
        Returns dict of face arrays of length N-1: {'D', 'Sr', 'tau'}.
        """
        N = len(r)
        nfaces = N - 1

        # Reconstruct primitive variables to faces (left/right states)
        (rhoL, vL, pL), (rhoR, vR, pR) = reconstructor.reconstruct_primitive_variables(
            rho0, vr, pressure, x=r, boundary_type="reflecting"
        )
        # Se asume que el reconstructor devuelve arrays de longitud nfaces.
        # Si devolviera longitud N, adapta aquí con slicing [:-1]/[1:].

        # Geometric quantities at faces (averaged)
        alpha_f   = 0.5 * (g['alpha'][:-1]   + g['alpha'][1:])
        beta_r_f  = 0.5 * (g['beta_r'][:-1]  + g['beta_r'][1:])
        gamma_rr_f = 0.5 * (g['gamma_rr'][:-1] + g['gamma_rr'][1:])
        J_f       = g['J_face']
        sqrt_gh_f = g['sqrt_g_hat_face']

        # Apply physical limiters if available
        if hasattr(reconstructor, "apply_physical_limiters"):
            (rhoL, vL, pL), (rhoR, vR, pR) = reconstructor.apply_physical_limiters(
                (rhoL, vL, pL), (rhoR, vR, pR),
                atmosphere_rho=1e-13, p_floor=1e-15, v_max=0.999999,
                gamma_rr=gamma_rr_f
            )

        # Convert primitives to conservatives at faces
        UL_D, UL_Sr, UL_tau = prim_to_cons(rhoL, vL, pL, gamma_rr_f, eos)
        UR_D, UR_Sr, UR_tau = prim_to_cons(rhoR, vR, pR, gamma_rr_f, eos)

        # Batch format for Riemann solver  -> physical fluxes F_phys (length nfaces)
        UL_batch    = np.stack([UL_D, UL_Sr, UL_tau], axis=1)
        UR_batch    = np.stack([UR_D, UR_Sr, UR_tau], axis=1)
        primL_batch = np.stack([rhoL, vL, pL], axis=1)
        primR_batch = np.stack([rhoR, vR, pR], axis=1)

        F_phys_batch = riemann_solver.solve_batch(
            UL_batch, UR_batch, primL_batch, primR_batch,
            gamma_rr_f, alpha_f, beta_r_f, eos
        )

        # Densitize: √ĝ (α J) F_phys
        dens_factor = sqrt_gh_f * alpha_f * J_f                       # length nfaces
        F_hat_batch = dens_factor[:, None] * F_phys_batch             # (nfaces, 3)

        # Extract components (faces)
        flux_D   = F_hat_batch[:, 0]
        flux_Sr  = F_hat_batch[:, 1]
        flux_tau = F_hat_batch[:, 2]

        # Sanity (optional)
        # assert flux_D.shape[0] == nfaces

        return {'D': flux_D, 'Sr': flux_Sr, 'tau': flux_tau}

    # ============================================================================
    # UTILITY METHODS
    # ============================================================================

    def _compute_flux_divergence(self, flux_face, sqrt_g_cell, dr, grid):
        """
        Compute flux divergence using Engrenage operator:
            div_hat ≡ -(1/√ĝ) ∂_r[ flux_face ]
        where 'flux_face' = √ĝ (α J) F_phys at faces (length N-1),
        and sqrt_g_cell is √ĝ at cell centers (length N).
        Returns array for interior cells (length N-2).
        """
        # High-order operator from grid if available
        if hasattr(grid, 'derivs'):
            # Promote face fluxes to cell-centered by arithmetic average
            flux_cell = 0.5 * (flux_face[:-1] + flux_face[1:])   # length N-2
            flux_full = np.zeros(len(sqrt_g_cell))               # length N
            flux_full[1:-1] = flux_cell
            # Apply derivative matrix (assumed to act on cell-centered arrays)
            dflux_dr = grid.derivs.drn_matrix[1] @ flux_full
            return -dflux_dr[1:-1] / (sqrt_g_cell[1:-1] + 1e-30)
        else:
            # 2nd-order fallback using faces directly
            return -(np.diff(flux_face)) / (sqrt_g_cell[1:-1] * dr + 1e-30)

    def _compute_stress_energy_tensor(self, rho0, vr, pressure, W, h,
                                      alpha, beta_r, gamma_rr):
        """
        Compute stress-energy tensor components for perfect fluid.
        T^{μν} = ρ₀ h u^μ u^ν + p g^{μν}
        Returns: (T^{00}, T^{0r}, T^{rr})
        """
        # 3+1 inverse metric components
        g00 = -1.0 / (alpha ** 2)
        g0r = beta_r / (alpha ** 2)
        grr = 1.0 / gamma_rr - (beta_r ** 2) / (alpha ** 2)

        # 4-velocity
        ut = W / alpha
        ur = W * (vr - beta_r / alpha)

        # Stress-energy tensor (mixed indices as needed later)
        T00 = rho0 * h * ut * ut + pressure * g00
        T0r = rho0 * h * ut * ur + pressure * g0r
        Trr = rho0 * h * ur * ur + pressure * grr

        return T00, T0r, Trr

    def _compute_Dhat_beta(self, bssn_vars, bssn_d1, background, r):
        """
        Compute D̂_r β^j = ∂_r β^j + Γ̂^j_{r k} β^k.
        Returns array of shape (N, SPACEDIM) for the radial derivative.
        """
        N = len(r)
        Dhat_beta = np.zeros((N, SPACEDIM))

        if not hasattr(bssn_d1, 'shift_U') or bssn_d1.shift_U is None:
            return Dhat_beta

        shift_d1 = np.asarray(bssn_d1.shift_U)   # ∂_i β^j, shape (N, SPACEDIM, SPACEDIM)
        shift_U  = np.asarray(bssn_vars.shift_U) # β^k,   shape (N, SPACEDIM)
        hat_chris = background.hat_christoffel   # Γ̂^i_{jk}, shape (N, SPACEDIM, SPACEDIM, SPACEDIM)

        if shift_d1.ndim >= 3 and shift_d1.shape[2] > i_r:
            # D̂_r β^j = ∂_r β^j + Γ̂^j_{r k} β^k
            for j in range(SPACEDIM):
                Dhat_beta[:, j]  = shift_d1[:, j, i_r]  # ∂_r β^j
                # Contract over k: Γ̂^j_{r k} β^k
                Dhat_beta[:, j] += np.einsum('xk,xk->x', shift_U, hat_chris[:, j, i_r, :])

        return Dhat_beta

    def _compute_Dhat_gamma(self, bssn_vars, bssn_d1, background, r,
                            phi, bar_gamma_LL):
        """
        Compute D̂_r γ_{jk} using:
            D̂_i γ_{jk} = e^{4φ} [4 γ̄_{jk} ∂_i φ + D̂_i γ̄_{jk}]
        Returns array (N, SPACEDIM, SPACEDIM) for radial derivative.
        """
        N = len(r)
        e4phi = np.exp(4.0 * phi)

        # D̂_i γ̄_{jk} from tensoralgebra (hat-covariant derivative)
        hat_D_bar_gamma = get_hat_D_bar_gamma_LL(
            r, bssn_vars.h_LL, bssn_d1.h_LL, background
        )  # shape (N, SPACEDIM, SPACEDIM, SPACEDIM)

        # ∂_r φ
        dphi_dr = np.zeros(N)
        if hasattr(bssn_d1, 'phi') and bssn_d1.phi is not None:
            phi_der = np.asarray(bssn_d1.phi)
            if phi_der.ndim >= 2:
                dphi_dr = phi_der[:, i_r]

        # D̂_r γ_{jk} = e^{4φ} [4 γ̄_{jk} ∂_r φ + D̂_r γ̄_{jk}]
        Dhat_gamma = np.zeros((N, SPACEDIM, SPACEDIM))
        for j in range(SPACEDIM):
            for k in range(SPACEDIM):
                Dhat_bar_gamma_jk = hat_D_bar_gamma[:, j, k, i_r]
                Dhat_gamma[:, j, k] = e4phi * (
                    4.0 * bar_gamma_LL[:, j, k] * dphi_dr + Dhat_bar_gamma_jk
                )

        return Dhat_gamma

    def _get_mesh_spacing(self, grid, r):
        """Get mesh spacing dr."""
        if hasattr(grid, 'derivs') and hasattr(grid.derivs, 'dx'):
            return float(grid.derivs.dx)
        elif hasattr(grid, 'dr'):
            return float(grid.dr)
        else:
            return float(r[1] - r[0]) if len(r) > 1 else 1.0

    def _apply_ghost_cell_boundaries(self, D, Sr, tau, r):
        """
        Apply boundary conditions to conservative variables.

        At origin (r≈0): Parity conditions
            D, τ: even (symmetric)
            Sr: odd (antisymmetric)

        At outer boundary: Extrapolation
        """
        N = len(r)

        if NUM_GHOSTS > 0:
            # Inner boundary: parity
            mir_slice = slice(2 * NUM_GHOSTS - 1, NUM_GHOSTS - 1, -1)
            D[:NUM_GHOSTS]   = D[mir_slice]
            Sr[:NUM_GHOSTS]  = -Sr[mir_slice]
            tau[:NUM_GHOSTS] = tau[mir_slice]

            # Outer boundary: extrapolation
            last_interior = N - NUM_GHOSTS - 1
            if last_interior >= 0:
                D[-NUM_GHOSTS:]   = D[last_interior]
                Sr[-NUM_GHOSTS:]  = Sr[last_interior]
                tau[-NUM_GHOSTS:] = tau[last_interior]

        return D, Sr, tau

    def _apply_rhs_boundary_conditions(self, rhs_D, rhs_Sr, rhs_tau, r):
        """Apply boundary conditions to RHS."""
        N = len(r)

        if NUM_GHOSTS > 0:
            # Inner boundary: parity
            mir_slice = slice(2 * NUM_GHOSTS - 1, NUM_GHOSTS - 1, -1)
            rhs_D[:NUM_GHOSTS]   = rhs_D[mir_slice]
            rhs_Sr[:NUM_GHOSTS]  = -rhs_Sr[mir_slice]
            rhs_tau[:NUM_GHOSTS] = rhs_tau[mir_slice]

            # Outer boundary: extrapolation
            last_interior = N - NUM_GHOSTS - 1
            if last_interior >= 0:
                rhs_D[-NUM_GHOSTS:]   = rhs_D[last_interior]
                rhs_Sr[-NUM_GHOSTS:]  = rhs_Sr[last_interior]
                rhs_tau[-NUM_GHOSTS:] = rhs_tau[last_interior]

        return rhs_D, rhs_Sr, rhs_tau
    





















# valencia_reference_metric.py
"""
Valencia with reference metric - corrected minimal version.
"""

import numpy as np

from source.bssn.tensoralgebra import (
    SPACEDIM,
    get_bar_gamma_LL,
    get_det_bar_gamma,
    get_bar_A_LL,
    get_hat_D_bar_gamma_LL
)
from source.backgrounds.sphericalbackground import i_r, i_t, i_p
from source.core.spacing import NUM_GHOSTS
from source.matter.hydro.cons2prim import prim_to_cons


class ValenciaReferenceMetric:
    """Valencia with reference metric - Engrenage-compatible implementation."""

    def compute_rhs(self, D, Sr, tau, rho0, vr, pressure, W, h,
                    r, bssn_vars, bssn_d1, background, spacetime_mode,
                    eos, grid, reconstructor, riemann_solver):
        """
        RHS computation:
        - Engrenage divergence captures most geometry automatically
        - Only add pressure-geometric term for momentum
        - Sources from spacetime curvature
        """
        
        dr = self._get_mesh_spacing(grid, r)
        N = len(r)
        
        D, Sr, tau = D.copy(), Sr.copy(), tau.copy()
        D, Sr, tau = self._apply_ghost_cell_boundaries(D, Sr, tau, r)
        
        g = self._extract_geometry(r, bssn_vars, spacetime_mode, background)
        
        flux_hat = self._compute_interface_fluxes(
            rho0, vr, pressure, g, r, eos, reconstructor, riemann_solver
        )
        
        # ====================================================================
        # ENGRENAGE DIVERGENCE (captures most connections automatically)
        # ====================================================================
        
        gh_cell = g['sqrt_g_hat_cell']
        div_D = -(np.diff(flux_hat['D'])) / (gh_cell[1:-1] * dr + 1e-30)
        div_Sr = -(np.diff(flux_hat['Sr'])) / (gh_cell[1:-1] * dr + 1e-30)
        div_tau = -(np.diff(flux_hat['tau'])) / (gh_cell[1:-1] * dr + 1e-30)
        
        # ====================================================================
        # MOMENTUM CONNECTION (pressure-geometric coupling)
        # ====================================================================
        # This term comes from the mixed-index tensor nature of (f_S)^j_i
        
        det_hat = np.asarray(background.det_hat_gamma, dtype=float)
        ddet_dr = np.asarray(background.d1_det_hat_gamma, dtype=float)[:, i_r]
        Gamma_trace_r = 0.5 * ddet_dr / (det_hat + 1e-30)  # Γ̂^k_{rk}
        
        # Add pressure-geometric term to momentum
        conn_Sr = (g['alpha'][1:-1] * g['J_cell'][1:-1] * 
                   pressure[1:-1] * Gamma_trace_r[1:-1])
        
        # ====================================================================
        # PHYSICAL SOURCES
        # ====================================================================
        
        src_Sr, src_tau = self._compute_source_terms(
            rho0, vr, pressure, W, h, g, bssn_vars, bssn_d1,
            background, spacetime_mode, r
        )
        
        # ====================================================================
        # ASSEMBLE
        # ====================================================================
        
        rhs_D = np.zeros(N)
        rhs_Sr = np.zeros(N)
        rhs_tau = np.zeros(N)
        
        rhs_D[1:-1] = div_D
        rhs_Sr[1:-1] = div_Sr + conn_Sr + src_Sr[1:-1]
        rhs_tau[1:-1] = div_tau + src_tau[1:-1]
        
        J = g['J_cell'] + 1e-30
        rhs_D /= J
        rhs_Sr /= J
        rhs_tau /= J
        
        rhs_D, rhs_Sr, rhs_tau = self._apply_rhs_boundary_conditions(
            rhs_D, rhs_Sr, rhs_tau, r
        )
        
        return rhs_D, rhs_Sr, rhs_tau

    # ========================================================================
    # SOURCES (general with einsum, following BSSN style)
    # ========================================================================
    
    def _compute_source_terms(self, rho0, vr, pressure, W, h, g,
                              bssn_vars, bssn_d1, background, spacetime_mode, r):
        """Physical sources - general implementation with einsum."""
        
        N = len(r)
        J = g['J_cell']
        
        if spacetime_mode == "fixed_minkowski":
            return np.zeros(N), np.zeros(N)
        
        alpha, beta_r, gamma_rr = g['alpha'], g['beta_r'], g['gamma_rr']
        
        # Stress-energy tensor
        T00, T0r, Trr = self._compute_stress_energy_tensor(
            rho0, vr, pressure, W, h, alpha, beta_r, gamma_rr
        )
        
        # Build 3D vectors for contractions
        beta_U = np.zeros((N, SPACEDIM))
        beta_U[:, i_r] = beta_r
        
        T0U = np.zeros((N, SPACEDIM))
        T0U[:, i_r] = T0r
        
        # Full spatial metric
        phi = np.asarray(bssn_vars.phi, dtype=float)
        e4phi = np.exp(4.0 * phi)
        bar_gamma_LL = get_bar_gamma_LL(r, bssn_vars.h_LL, background)
        gamma_LL = e4phi[:, None, None] * bar_gamma_LL
        gamma_UU = np.linalg.inv(gamma_LL)
        
        # T^{ij} tensor
        TUU = np.zeros((N, SPACEDIM, SPACEDIM))
        TUU[:, i_r, i_r] = Trr
        TUU[:, i_t, i_t] = pressure * gamma_UU[:, i_t, i_t]
        TUU[:, i_p, i_p] = pressure * gamma_UU[:, i_p, i_p]
        
        # T^{0}_i with proper lowering
        beta_lower = np.einsum('xij,xj->xi', gamma_LL, beta_U)
        T0_lower = T00[:, None] * beta_lower + np.einsum('xj,xij->xi', T0U, gamma_LL)
        
        # Geometric derivatives
        dalpha_dx = np.zeros((N, SPACEDIM))
        if hasattr(bssn_d1, 'lapse') and bssn_d1.lapse is not None:
            dalpha_dx = np.asarray(bssn_d1.lapse)
        
        # D̂_i β^j
        Dhat_beta_U = np.zeros((N, SPACEDIM, SPACEDIM))
        if hasattr(bssn_d1, 'shift_U') and bssn_d1.shift_U is not None:
            shift_d1 = np.asarray(bssn_d1.shift_U)
            if shift_d1.ndim >= 3:
                Dhat_beta_U = shift_d1.copy()
                hat_chris = background.hat_christoffel
                Dhat_beta_U += np.einsum('xjik,xk->xij', hat_chris, beta_U)
        
        # D̂_i γ_{jk}
        dphi_dx = np.zeros((N, SPACEDIM))
        if hasattr(bssn_d1, 'phi') and bssn_d1.phi is not None:
            dphi_dx = np.asarray(bssn_d1.phi)
        
        hat_D_bar_gamma = get_hat_D_bar_gamma_LL(r, bssn_vars.h_LL,
                                                  bssn_d1.h_LL, background)
        
        Dhat_gamma_LL = np.zeros((N, SPACEDIM, SPACEDIM, SPACEDIM))
        for i in range(SPACEDIM):
            for j in range(SPACEDIM):
                for k in range(SPACEDIM):
                    Dhat_gamma_LL[:, i, j, k] = e4phi * (
                        4.0 * bar_gamma_LL[:, j, k] * dphi_dx[:, i]
                        + hat_D_bar_gamma[:, j, k, i]
                    )
        
        # K_{ij}
        K = np.asarray(bssn_vars.K, dtype=float)
        bar_A_LL = get_bar_A_LL(r, bssn_vars, background)
        K_LL = e4phi[:, None, None] * bar_A_LL + (K / 3.0)[:, None, None] * gamma_LL
        
        # Tensor block
        tensor_block = (
            T00[:, None, None] * np.einsum('xi,xj->xij', beta_U, beta_U)
            + 2.0 * np.einsum('xi,xj->xij', T0U, beta_U)
            + TUU
        )
        
        # SOURCES with einsum
        src_Sr_vector = J[:, None] * alpha[:, None] * (
            -T00[:, None] * alpha[:, None] * dalpha_dx
            + np.einsum('xi,xij->xi', T0_lower, Dhat_beta_U)
            + 0.5 * np.einsum('xjk,xijk->xi', tensor_block, Dhat_gamma_LL)
        )
        
        src_tau = J * alpha * (
            np.einsum('x,xi,xj,xij->x', T00, beta_U, beta_U, K_LL)
            - np.einsum('x,xi,xi->x', T00, beta_U, dalpha_dx)
            + 2.0 * np.einsum('xi,xj,xij->x', T0U, beta_U, K_LL)
            - np.einsum('xi,xi->x', T0U, dalpha_dx)
            + np.einsum('xij,xij->x', TUU, K_LL)
        )
        
        return src_Sr_vector[:, i_r], src_tau

    # ========================================================================
    # GEOMETRY & UTILITIES (same as before)
    # ========================================================================
    
    def _extract_geometry(self, r, bssn_vars, spacetime_mode, background):
        N = len(r)
        g = {}
        
        if spacetime_mode == "fixed_minkowski":
            g['alpha'] = np.ones(N)
            g['beta_r'] = np.zeros(N)
            g['e6phi'] = np.ones(N)
            g['gamma_rr'] = np.ones(N)
        else:
            g['alpha'] = np.asarray(bssn_vars.lapse, dtype=float)
            
            if hasattr(bssn_vars, 'shift_U') and bssn_vars.shift_U is not None:
                shift_array = np.asarray(bssn_vars.shift_U)
                g['beta_r'] = (shift_array[:, i_r].astype(float)
                              if shift_array.ndim >= 2 else np.zeros(N))
            else:
                g['beta_r'] = np.zeros(N)
            
            phi_arr = np.asarray(bssn_vars.phi, dtype=float)
            g['e6phi'] = np.exp(6.0 * phi_arr)
            
            bar_gamma_LL = get_bar_gamma_LL(r, bssn_vars.h_LL, background)
            e4phi = np.exp(4.0 * phi_arr)
            g['gamma_rr'] = e4phi * bar_gamma_LL[:, i_r, i_r]
        
        r_arr = np.asarray(r, dtype=float)
        r_face = 0.5 * (r_arr[:-1] + r_arr[1:])
        g['sqrt_g_hat_cell'] = np.maximum(np.abs(r_arr), 1e-30) ** 2
        g['sqrt_g_hat_face'] = np.maximum(np.abs(r_face), 1e-30) ** 2
        
        if spacetime_mode != "fixed_minkowski":
            det_bar_gamma = get_det_bar_gamma(r, bssn_vars.h_LL, background)
            sqrt_bar_gamma = np.sqrt(np.abs(det_bar_gamma) + 1e-30)
            sqrt_hat_gamma = np.sqrt(np.abs(background.det_hat_gamma) + 1e-30)
            J_cell = g['e6phi'] * sqrt_bar_gamma / sqrt_hat_gamma
        else:
            J_cell = np.ones(N)
        
        g['J_cell'] = J_cell
        g['J_face'] = 0.5 * (J_cell[:-1] + J_cell[1:])
        
        return g
    
    def _compute_interface_fluxes(self, rho0, vr, pressure, g, r,
                                   eos, reconstructor, riemann_solver):
        N = len(r)
        
        (rhoL, vL, pL), (rhoR, vR, pR) = reconstructor.reconstruct_primitive_variables(
            rho0, vr, pressure, x=r, boundary_type="reflecting"
        )
        
        rhoL, vL, pL = rhoL[1:-1], vL[1:-1], pL[1:-1]
        rhoR, vR, pR = rhoR[1:-1], vR[1:-1], pR[1:-1]
        
        alpha_f = 0.5 * (g['alpha'][:-1] + g['alpha'][1:])
        beta_r_f = 0.5 * (g['beta_r'][:-1] + g['beta_r'][1:])
        gamma_rr_f = 0.5 * (g['gamma_rr'][:-1] + g['gamma_rr'][1:])
        J_f = g['J_face']
        sqrt_gh_f = g['sqrt_g_hat_face']
        
        if hasattr(reconstructor, "apply_physical_limiters"):
            (rhoL, vL, pL), (rhoR, vR, pR) = reconstructor.apply_physical_limiters(
                (rhoL, vL, pL), (rhoR, vR, pR),
                atmosphere_rho=1e-13, p_floor=1e-15, v_max=0.999999,
                gamma_rr=gamma_rr_f
            )
        
        UL_D, UL_Sr, UL_tau = prim_to_cons(rhoL, vL, pL, gamma_rr_f, eos)
        UR_D, UR_Sr, UR_tau = prim_to_cons(rhoR, vR, pR, gamma_rr_f, eos)
        
        UL_batch = np.stack([UL_D, UL_Sr, UL_tau], axis=1)
        UR_batch = np.stack([UR_D, UR_Sr, UR_tau], axis=1)
        primL_batch = np.stack([rhoL, vL, pL], axis=1)
        primR_batch = np.stack([rhoR, vR, pR], axis=1)
        
        F_phys_batch = riemann_solver.solve_batch(
            UL_batch, UR_batch, primL_batch, primR_batch,
            gamma_rr_f, alpha_f, beta_r_f, eos
        )
        
        dens_factor = sqrt_gh_f * alpha_f * J_f
        F_hat_batch = dens_factor[:, None] * F_phys_batch
        
        return {
            'D': F_hat_batch[:, 0],
            'Sr': F_hat_batch[:, 1],
            'tau': F_hat_batch[:, 2]
        }
    
    def _compute_stress_energy_tensor(self, rho0, vr, pressure, W, h,
                                      alpha, beta_r, gamma_rr):
        grr = 1.0 / gamma_rr
        beta_u = grr * beta_r
        ut = W / alpha
        ur = W * (vr - beta_u / alpha)
        
        g00 = -1.0 / (alpha ** 2)
        g0r = beta_u / (alpha ** 2)
        grr_eff = grr - (beta_u ** 2) / (alpha ** 2)
        
        T00 = rho0 * h * ut * ut + pressure * g00
        T0r = rho0 * h * ut * ur + pressure * g0r
        Trr = rho0 * h * ur * ur + pressure * grr_eff
        
        return T00, T0r, Trr
    
    def _get_mesh_spacing(self, grid, r):
        if hasattr(grid, 'derivs') and hasattr(grid.derivs, 'dx'):
            return float(grid.derivs.dx)
        elif hasattr(grid, 'dr'):
            return float(grid.dr)
        return float(r[1] - r[0]) if len(r) > 1 else 1.0
    
    def _apply_ghost_cell_boundaries(self, D, Sr, tau, r):
        N = len(r)
        if NUM_GHOSTS > 0:
            mir = slice(2 * NUM_GHOSTS - 1, NUM_GHOSTS - 1, -1)
            D[:NUM_GHOSTS], Sr[:NUM_GHOSTS], tau[:NUM_GHOSTS] = D[mir], -Sr[mir], tau[mir]
            last = N - NUM_GHOSTS - 1
            if last >= 0:
                D[-NUM_GHOSTS:], Sr[-NUM_GHOSTS:], tau[-NUM_GHOSTS:] = D[last], Sr[last], tau[last]
        return D, Sr, tau
    
    def _apply_rhs_boundary_conditions(self, rhs_D, rhs_Sr, rhs_tau, r):
        N = len(r)
        if NUM_GHOSTS > 0:
            mir = slice(2 * NUM_GHOSTS - 1, NUM_GHOSTS - 1, -1)
            rhs_D[:NUM_GHOSTS], rhs_Sr[:NUM_GHOSTS], rhs_tau[:NUM_GHOSTS] = rhs_D[mir], -rhs_Sr[mir], rhs_tau[mir]
            last = N - NUM_GHOSTS - 1
            if last >= 0:
                rhs_D[-NUM_GHOSTS:], rhs_Sr[-NUM_GHOSTS:], rhs_tau[-NUM_GHOSTS:] = rhs_D[last], rhs_Sr[last], rhs_tau[last]
        return rhs_D, rhs_Sr, rhs_tau

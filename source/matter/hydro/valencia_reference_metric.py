# valencia_reference_metric.py
"""
Valencia formulation with reference metric - full 3D BSSN-style implementation.

Follows the exact same tensor algebra pattern as bssnrhs.py:
- Full 3D einsum contractions for all tensor operations
- Spherical symmetry imposed by velocity (v^θ = v^φ = 0) and background metric
- Source terms following NRPy+ GRHD_equations.py structure
- Connection terms with CORRECT signs for covariant divergence

Conservative evolution equations in curved coordinates:
    ∂_t(U) + (1/√ĝ) ∂_j[√ĝ F^j] = S + S_connection

Where:
    U = (D, S_i, τ)  conserved variables
    F^j = physical fluxes
    S = geometric source terms
    S_connection = -Γ̂^k_{jk} F^j (covariant divergence correction)
"""

import numpy as np

from source.bssn.tensoralgebra import (
    SPACEDIM,
    get_bar_gamma_LL,
    get_bar_gamma_UU,
    get_det_bar_gamma,
    get_bar_A_LL,
    get_hat_D_bar_gamma_LL,
)
from source.backgrounds.sphericalbackground import i_r, i_t, i_p
from source.core.spacing import NUM_GHOSTS
from source.matter.hydro.cons2prim import prim_to_cons


class ValenciaReferenceMetric:
    """Valencia formulation - full 3D tensor algebra following BSSN pattern."""

    def __init__(self, boundary_mode="parity", *, atmosphere_rho: float = 1e-13,
                 p_floor: float = 1e-15, v_max: float = 0.999999):
        """
        Initialize Valencia formulation.

        Parameters:
        -----------
        boundary_mode : str
            "parity" - Parity boundary conditions at inner boundary (r=0)
            "outflow" - Outflow (zero-gradient) at both boundaries
        atmosphere_rho : float
            Atmosphere density floor used to stabilize reconstruction near the surface.
        p_floor : float
            Minimum pressure used by physical limiters at interfaces.
        v_max : float
            Maximum allowed |v^r| used by physical limiters at interfaces.
        """
        self.boundary_mode = boundary_mode
        # Near-surface stabilization parameters (kept in sync with PerfectFluid/cons2prim)
        self.atmosphere_rho = float(atmosphere_rho)
        self.p_floor = float(p_floor)
        self.v_max = float(v_max)

    def compute_rhs(self, D, Sr, tau, rho0, vr, pressure, W, h,
                    r, bssn_vars, bssn_d1, background, spacetime_mode,
                    eos, grid, reconstructor, riemann_solver):
        """
        Compute RHS of Valencia equations using full 3D tensor contractions.

        Follows bssnrhs.py structure with einsum for all tensor operations.
        Implements equations from NRPy+ GRHD_equations.py with CORRECT signs.
        """
        dr = self._get_mesh_spacing(grid, r)
        N = len(r)

        # Compute W and h from primitives if not provided
        if W is None:
            v2 = vr * vr
            W = 1.0 / np.sqrt(np.maximum(1.0 - v2, 1e-16))

        if h is None:
            eps = eos.eps_from_rho_p(rho0, pressure)
            h = 1.0 + eps + pressure / np.maximum(rho0, 1e-30)

        # Copy and apply boundary conditions to conservatives
        D, Sr, tau = D.copy(), Sr.copy(), tau.copy()
        # Apply only inner (parity) BC to conservatives in "parity" mode to avoid
        # contradicting primitive-based BC at the outer boundary. In "outflow" mode
        # keep zero-gradient on both ends for robustness.
        D, Sr, tau = self._apply_ghost_cell_boundaries(D, Sr, tau, r)

        # Extract geometry (α, β^i, γ_ij, √γ, √ĝ, etc.)
        g = self._extract_geometry(r, bssn_vars, spacetime_mode, background)

        # Compute densitized fluxes at interfaces: √ĝ_face (α J F_phys)
        flux_hat = self._compute_interface_fluxes(
            rho0, vr, pressure, g, r, eos, reconstructor, riemann_solver
        )

        # Reference metric Christoffel symbols Γ̂^i_{jk}
        hat_chris = background.hat_christoffel  # (N, 3, 3, 3)

        # Densitized fluxes at faces
        F_D_face = flux_hat['D']
        F_Sr_face = flux_hat['Sr']
        F_tau_face = flux_hat['tau']

        # sqrt(ĝ) at cells (for normalization)
        sqrt_g_hat_cell = g['sqrt_g_hat_cell']

        # ====================================================================
        # FLUX DIVERGENCE: -(1/√ĝ) ∂_r[√ĝ F^r]
        # ====================================================================

        rhs_D = np.zeros(N)
        rhs_Sr = np.zeros(N)
        rhs_tau = np.zeros(N)

        # Compute divergence in interior cells
        for i in range(NUM_GHOSTS, N - NUM_GHOSTS):
            inv_vol = 1.0 / (sqrt_g_hat_cell[i] * dr + 1e-30)

            rhs_D[i] = -(F_D_face[i] - F_D_face[i-1]) * inv_vol
            rhs_Sr[i] = -(F_Sr_face[i] - F_Sr_face[i-1]) * inv_vol
            rhs_tau[i] = -(F_tau_face[i] - F_tau_face[i-1]) * inv_vol

        # ====================================================================
        # CONNECTION TERMS: -Γ̂^k_{jk} F^j (covariant divergence correction)
        # ====================================================================
        # This is the difference between flat and curved space divergence
        # Note: NEGATIVE sign is correct (see NRPy+ line 367-375 with RHS assembly)
        # Only compute for curved spacetime

        if spacetime_mode != "fixed_minkowski":
            # Interpolate densitized fluxes to cell centers for connection terms
            F_D_cell = np.zeros(N)
            F_Sr_cell = np.zeros(N)
            F_tau_cell = np.zeros(N)
            F_D_cell[1:-1] = 0.5 * (F_D_face[:-1] + F_D_face[1:])
            F_Sr_cell[1:-1] = 0.5 * (F_Sr_face[:-1] + F_Sr_face[1:])
            F_tau_cell[1:-1] = 0.5 * (F_tau_face[:-1] + F_tau_face[1:])

            # Build 3D flux vectors (extend 1D fluxes to 3D for einsum)
            fD_U = np.zeros((N, SPACEDIM))
            fD_U[:, i_r] = F_D_cell

            fTau_U = np.zeros((N, SPACEDIM))
            fTau_U[:, i_r] = F_tau_cell

            # For momentum, build full T^j_i flux tensor
            T4UD = self._compute_T4UD(rho0, vr, pressure, W, h, g, bssn_vars, background, r)

            alpha = g['alpha']
            sqrt_gamma = g['sqrt_gamma']
            sqrt_g_hat = g['sqrt_g_hat_cell']

            # Momentum flux tensor: F^j_i = √ĝ (α √γ T^j_i)
            fS_mixed = np.zeros((N, SPACEDIM, SPACEDIM))
            for j in range(SPACEDIM):
                for i in range(SPACEDIM):
                    fS_mixed[:, j, i] = sqrt_g_hat * alpha * sqrt_gamma * T4UD[:, j, i]

            # Connection terms with CORRECT negative sign
            # Density: -Γ̂^k_{jk} F^j_D / √ĝ
            Gamma_trace = np.einsum('xkkj->xj', hat_chris)  # Γ̂^k_{jk}
            conn_D = -np.einsum('xj,xj->x', Gamma_trace, fD_U) / (sqrt_g_hat + 1e-30)

            # Energy: -Γ̂^k_{jk} F^j_τ / √ĝ
            conn_tau = -np.einsum('xj,xj->x', Gamma_trace, fTau_U) / (sqrt_g_hat + 1e-30)

            # Momentum: (Γ̂^j_{jl} F^l_i - Γ̂^l_{ji} F^j_l) / √ĝ (NRPy+ line 383-386)
            # CORRECTO
            conn_S_vector = (
                -np.einsum('xjjl,xli->xi', hat_chris, fS_mixed) / (sqrt_g_hat[:, None] + 1e-30)
                + np.einsum('xlji,xjl->xi', hat_chris, fS_mixed) / (sqrt_g_hat[:, None] + 1e-30)
            )
            conn_Sr = conn_S_vector[:, i_r]

            # Add connection terms to RHS
            rhs_D += conn_D
            rhs_tau += conn_tau
            rhs_Sr += conn_Sr

        # ====================================================================
        # SOURCE TERMS (geometric couplings - NRPy+ style)
        # ====================================================================

        src_Sr, src_tau = self._compute_source_terms(
            rho0, vr, pressure, W, h, g, bssn_vars, bssn_d1,
            background, spacetime_mode, r
        )

        rhs_Sr += src_Sr
        rhs_tau += src_tau

        # ====================================================================
        # FINAL NORMALIZATION
        # ====================================================================
        # The RHS is already properly normalized by the (1/√ĝ) factors above
        # No additional division needed

        # DISABLED: BC on RHS are unnecessary in Method of Lines
        # The state BC are sufficient - RHS BC were redundant/harmful
        # rhs_D, rhs_Sr, rhs_tau = self._apply_rhs_boundary_conditions(
        #     rhs_D, rhs_Sr, rhs_tau, r
        # )

        # NOTE: No RHS limiter applied following GRoovy/NRPy+ approach
        # Atmosphere handling is done via reconstruction limiters and cons2prim floors

        return rhs_D, rhs_Sr, rhs_tau

    def _compute_T4UD(self, rho0, vr, pressure, W, h, g, bssn_vars, background, r):
        """
        Compute T^i_j = T^{ik} γ_{kj} following NRPy+ (lines 176-213).

        This is the spatial part of the stress-energy tensor with one index raised
        and one lowered, needed for momentum flux computation.

        Returns: T^i_j as (N, 3, 3) array
        """
        N = len(rho0)

        # Build 4-velocity components
        alpha = g['alpha']

        # u^t = W / α (timelike component for Valencia)
        ut = W / alpha

        # u^i = W v^i (spatial components, only radial in spherical symmetry)
        uU = np.zeros((N, SPACEDIM))
        uU[:, i_r] = W * vr

        # Build full 3D metric γ_ij = e^{4φ} γ̄_ij (BSSN conformal decomposition)
        phi = np.asarray(bssn_vars.phi, dtype=float)
        e4phi = np.exp(4.0 * phi)
        bar_gamma_LL = get_bar_gamma_LL(r, bssn_vars.h_LL, background)
        gamma_LL = e4phi[:, None, None] * bar_gamma_LL
        gamma_UU = np.linalg.inv(gamma_LL)

        # Build contravariant spatial stress-energy: T^{ij} = ρ₀ h u^i u^j + P γ^{ij}
        # (NRPy+ line 159-161)
        TUU_spatial = np.zeros((N, SPACEDIM, SPACEDIM))
        for i in range(SPACEDIM):
            for j in range(SPACEDIM):
                TUU_spatial[:, i, j] = (
                    rho0 * h * uU[:, i] * uU[:, j]
                    + pressure * gamma_UU[:, i, j]
                )

        # Lower second index: T^i_j = T^{ik} γ_{kj} (NRPy+ line 194-198)
        # Correct contraction over k only, preserving j index
        T4UD = np.einsum('xik,xkj->xij', TUU_spatial, gamma_LL)

        return T4UD

    def _compute_source_terms(self, rho0, vr, pressure, W, h, g,
                              bssn_vars, bssn_d1, background, spacetime_mode, r):
        """
        Physical source terms following NRPy+ GRHD_equations.py (lines 334-438).

        Energy source (tau_source_term, NRPy+ line 334-358):
            S_τ = α √γ [K_ij (T^tt β^i β^j + 2 T^{ti} β^j + T^{ij})
                        - (T^tt β^i + T^{ti}) ∂_i α]

        Momentum source (S_tilde_source_termD, NRPy+ line 388-438):
            S_{S_i} = α √γ [-T^tt α ∂_i α
                            + T^t_j (∂_i β^j + Γ̂^j_{ik} β^k)
                            + (1/2) T^{jk} ∇̂_i γ_{jk}]
        """
        N = len(r)

        if spacetime_mode == "fixed_minkowski":
            # In Minkowski, only hoop stress for momentum
            src_Sr = 2.0 * pressure / (r + 1e-30)
            src_tau = np.zeros(N)
            return src_Sr, src_tau

        alpha = g['alpha']
        sqrt_gamma = g['sqrt_gamma']

        # Build 3D shift vector β^i
        beta_U = np.zeros((N, SPACEDIM))
        beta_U[:, i_r] = g['beta_r']

        # Build full 3D metric γ_ij = e^{4φ} γ̄_ij
        phi = np.asarray(bssn_vars.phi, dtype=float)
        e4phi = np.exp(4.0 * phi)
        bar_gamma_LL = get_bar_gamma_LL(r, bssn_vars.h_LL, background)
        gamma_LL = e4phi[:, None, None] * bar_gamma_LL
        gamma_UU = np.linalg.inv(gamma_LL)

        # ====================================================================
        # BUILD STRESS-ENERGY TENSOR T^{μν} (contravariant)
        # ====================================================================

        # 4-velocity components
        ut = W / alpha
        uU = np.zeros((N, SPACEDIM))
        uU[:, i_r] = W * vr

        # T^{00} = ρ₀ h u^t u^t + P g^{tt} (NRPy+ line 159)
        # Note: g^{tt} = -1/α²
        T00 = rho0 * h * ut * ut - pressure / (alpha ** 2)

        # T^{0i} = ρ₀ h u^t u^i (NRPy+ line 159)
        T0U = np.zeros((N, SPACEDIM))
        T0U[:, i_r] = rho0 * h * ut * uU[:, i_r]

        # T^{ij} = ρ₀ h u^i u^j + P γ^{ij} (NRPy+ line 159-161)
        TUU = np.zeros((N, SPACEDIM, SPACEDIM))
        for i in range(SPACEDIM):
            for j in range(SPACEDIM):
                TUU[:, i, j] = (
                    rho0 * h * uU[:, i] * uU[:, j]
                    + pressure * gamma_UU[:, i, j]
                )

        # ====================================================================
        # GEOMETRIC QUANTITIES FOR SOURCE TERMS
        # ====================================================================

        # Lapse derivatives ∂_i α (NRPy+ line 340)
        dalpha_dx = np.zeros((N, SPACEDIM))
        if hasattr(bssn_d1, 'lapse') and bssn_d1.lapse is not None:
            dalpha_dx = np.asarray(bssn_d1.lapse)

        # Shift derivatives ∂_i β^j (NRPy+ line 393, 420)
        dbeta_dx = np.zeros((N, SPACEDIM, SPACEDIM))
        if hasattr(bssn_d1, 'shift_U') and bssn_d1.shift_U is not None:
            shift_d1 = np.asarray(bssn_d1.shift_U)
            if shift_d1.ndim >= 3:
                dbeta_dx = shift_d1.copy()

        # Covariant derivative of shift: ∇̂_i β^j = ∂_i β^j + Γ̂^j_{ik} β^k
        # (NRPy+ line 422-423)
        hat_chris = background.hat_christoffel
        hatD_beta_U = dbeta_dx + np.einsum('xjik,xk->xij', hat_chris, beta_U)

        # Covariant derivative of metric: ∇̂_i γ_{jk} (NRPy+ line 407-415)
        # ∇̂_i γ_{jk} = e^{-4φ} [4 γ̄_{jk} ∂_i φ + ∂_i γ̄_{jk} - Γ̂^l_{ij} γ̄_{lk} - Γ̂^l_{ik} γ̄_{jl}]
        # Note: Final result is then multiplied by e^{4φ} to get physical metric derivative
        dphi_dx = np.zeros((N, SPACEDIM))
        if hasattr(bssn_d1, 'phi') and bssn_d1.phi is not None:
            dphi_dx = np.asarray(bssn_d1.phi)

        hat_D_bar_gamma = get_hat_D_bar_gamma_LL(r, bssn_vars.h_LL, bssn_d1.h_LL, background)

        hatD_gamma_LL = np.zeros((N, SPACEDIM, SPACEDIM, SPACEDIM))
        for i in range(SPACEDIM):
            for j in range(SPACEDIM):
                for k in range(SPACEDIM):
                    # Start with partial derivatives and phi term
                    hatD_gamma_LL[:, i, j, k] = (
                        4.0 * bar_gamma_LL[:, j, k] * dphi_dx[:, i]
                        + hat_D_bar_gamma[:, j, k, i]
                    )
                    # Subtract Christoffel correction terms (NRPy+ line 412-415)
                    for l in range(SPACEDIM):
                        hatD_gamma_LL[:, i, j, k] -= (
                            hat_chris[:, l, i, j] * bar_gamma_LL[:, l, k]
                            + hat_chris[:, l, i, k] * bar_gamma_LL[:, j, l]
                        )
                    # Multiply by e^{4φ} to get physical metric derivative
                    hatD_gamma_LL[:, i, j, k] *= e4phi

        # Extrinsic curvature K_ij = e^{4φ} Ā_ij + (K/3) γ_ij (NRPy+ line 336)
        K = np.asarray(bssn_vars.K, dtype=float)
        bar_A_LL = get_bar_A_LL(r, bssn_vars, background)
        K_LL = e4phi[:, None, None] * bar_A_LL + (K / 3.0)[:, None, None] * gamma_LL

        # ====================================================================
        # ENERGY SOURCE TERM (NRPy+ line 334-358)
        # ====================================================================

        # Tensor block: T^tt β^i β^j + 2 T^{ti} β^j + T^{ij} (NRPy+ line 347-351)
        tensor_block = (
            T00[:, None, None] * np.einsum('xi,xj->xij', beta_U, beta_U)
            + 2.0 * np.einsum('xi,xj->xij', T0U, beta_U)
            + TUU
        )

        # S_τ = α √γ [K_ij * tensor_block - (T^tt β^i + T^{ti}) ∂_i α]
        # (NRPy+ line 343-358)
        src_tau = alpha * sqrt_gamma * (
            np.einsum('xij,xij->x', K_LL, tensor_block)
            - np.einsum('x,xi,xi->x', T00, beta_U, dalpha_dx)
            - np.einsum('xi,xi->x', T0U, dalpha_dx)
        )

        # ====================================================================
        # MOMENTUM SOURCE TERM (NRPy+ line 388-438)
        # ====================================================================

        # Term 1: -T^tt α ∂_i α (NRPy+ line 418)
        first_term = -T00[:, None] * alpha[:, None] * dalpha_dx

        # Term 2: T^t_j ∇̂_i β^j (NRPy+ line 420-423)
        # First lower index: T^t_j = T^{tk} γ_{kj}
        T0_lower = np.einsum('xk,xkj->xj', T0U, gamma_LL)
        second_term = np.einsum('xj,xij->xi', T0_lower, hatD_beta_U)

        # Term 3: (1/2) T^{jk} ∇̂_i γ_{jk} (NRPy+ line 425-433)
        third_term = 0.5 * np.einsum('xjk,xijk->xi', TUU, hatD_gamma_LL)

        # Combine all momentum source terms (NRPy+ line 436-438)
        src_S_vector = alpha[:, None] * sqrt_gamma[:, None] * (
            first_term + second_term + third_term
        )

        # Extract radial component
        src_Sr = src_S_vector[:, i_r]

        return src_Sr, src_tau

    def _extract_geometry(self, r, bssn_vars, spacetime_mode, background):
        """
        Extract geometric quantities from BSSN variables.

        Returns dictionary with all geometric quantities needed for RHS computation.
        """
        N = len(r)
        g = {}

        if spacetime_mode == "fixed_minkowski":
            g['alpha'] = np.ones(N)
            g['beta_r'] = np.zeros(N)
            g['gamma_rr'] = np.ones(N)
            g['sqrt_gamma'] = np.ones(N)
            g['sqrt_g_hat_cell'] = background.det_hat_gamma ** 0.5  # r² for spherical
            g['sqrt_g_hat_face'] = 0.5 * (g['sqrt_g_hat_cell'][:-1] + g['sqrt_g_hat_cell'][1:])
        else:
            # Lapse α
            g['alpha'] = np.asarray(bssn_vars.lapse, dtype=float)

            # Shift β^r
            if hasattr(bssn_vars, 'shift_U') and bssn_vars.shift_U is not None:
                shift_array = np.asarray(bssn_vars.shift_U)
                g['beta_r'] = (shift_array[:, i_r].astype(float)
                              if shift_array.ndim >= 2 else np.zeros(N))
            else:
                g['beta_r'] = np.zeros(N)

            # Conformal factor e^{φ}
            phi_arr = np.asarray(bssn_vars.phi, dtype=float)
            e6phi = np.exp(6.0 * phi_arr)
            e4phi = np.exp(4.0 * phi_arr)

            # Physical metric γ_rr = e^{4φ} γ̄_rr
            bar_gamma_LL = get_bar_gamma_LL(r, bssn_vars.h_LL, background)
            g['gamma_rr'] = e4phi * bar_gamma_LL[:, i_r, i_r]

            # √γ = e^{6φ} √(det γ̄) (NRPy+ uses e6phi, line 223, 244, 257)
            det_bar_gamma = get_det_bar_gamma(r, bssn_vars.h_LL, background)
            g['sqrt_gamma'] = e6phi * np.sqrt(np.abs(det_bar_gamma) + 1e-30)

            # √ĝ for reference metric
            g['sqrt_g_hat_cell'] = np.sqrt(np.abs(background.det_hat_gamma) + 1e-30)
            g['sqrt_g_hat_face'] = 0.5 * (g['sqrt_g_hat_cell'][:-1] + g['sqrt_g_hat_cell'][1:])

        return g

    def _compute_interface_fluxes(self, rho0, vr, pressure, g, r,
                                   eos, reconstructor, riemann_solver):
        """
        Compute densitized fluxes at cell interfaces: √ĝ_face (α √γ F_phys).

        Returns fluxes already multiplied by √ĝ α √γ.
        """
        N = len(r)

        # Reconstruct primitives to left/right states at interfaces.
        # Map Valencia boundary_mode to reconstructor boundary_type
        recon_boundary = "reflecting" if self.boundary_mode == "parity" else "outflow"
        (rhoL, vL, pL), (rhoR, vR, pR) = reconstructor.reconstruct_primitive_variables(
            rho0, vr, pressure, x=r, boundary_type=recon_boundary
        )

        # Extract interior faces (exclude ghost zones)
        rhoL, vL, pL = rhoL[1:-1], vL[1:-1], pL[1:-1]
        rhoR, vR, pR = rhoR[1:-1], vR[1:-1], pR[1:-1]

        # Interpolate geometry to faces
        alpha_f = 0.5 * (g['alpha'][:-1] + g['alpha'][1:])
        beta_r_f = 0.5 * (g['beta_r'][:-1] + g['beta_r'][1:])
        gamma_rr_f = 0.5 * (g['gamma_rr'][:-1] + g['gamma_rr'][1:])
        sqrt_gamma_f = 0.5 * (g['sqrt_gamma'][:-1] + g['sqrt_gamma'][1:])
        sqrt_g_hat_f = g['sqrt_g_hat_face']

        # Enforce a strictly reflecting condition at the first physical interior face
        # (between the first two interior cells) to avoid tiny asymmetries at r≈0.
        if recon_boundary == "reflecting" and (len(vL) > NUM_GHOSTS):
            k0 = NUM_GHOSTS  # interface between cells NUM_GHOSTS and NUM_GHOSTS+1 after [1:-1] trim
            if k0 < len(vL):
                vL[k0] = 0.0
                vR[k0] = 0.0
                rho_ref = rho0[NUM_GHOSTS]
                p_ref = pressure[NUM_GHOSTS]
                rhoL[k0] = rho_ref
                rhoR[k0] = rho_ref
                pL[k0] = p_ref
                pR[k0] = p_ref

        # Surface/atmosphere flattening: in interfaces where density is near the configured
        # atmosphere, switch to first-order (average) and force v=0 to avoid overshoots.
        # Use a threshold relative to the chosen atmosphere density.
        dens_floor = max(self.atmosphere_rho, 1e-20)
        thr = 30.0 * dens_floor
        if len(rhoL) > 0:
            import numpy as _np
            mask = (rhoL < thr) | (rhoR < thr)
            if _np.any(mask):
                rho_avg = 0.5 * (rhoL + rhoR)
                p_avg = _np.maximum(0.5 * (pL + pR), self.p_floor)
                rhoL[mask] = rho_avg[mask]
                rhoR[mask] = rho_avg[mask]
                pL[mask] = p_avg[mask]
                pR[mask] = p_avg[mask]
                vL[mask] = 0.0
                vR[mask] = 0.0

        # Apply physical limiters if available
        if hasattr(reconstructor, "apply_physical_limiters"):
            (rhoL, vL, pL), (rhoR, vR, pR) = reconstructor.apply_physical_limiters(
                (rhoL, vL, pL), (rhoR, vR, pR),
                atmosphere_rho=self.atmosphere_rho,
                p_floor=self.p_floor,
                v_max=self.v_max,
                gamma_rr=gamma_rr_f
            )

        # Convert primitives to conservatives at interfaces
        UL_D, UL_Sr, UL_tau = prim_to_cons(rhoL, vL, pL, gamma_rr_f, eos)
        UR_D, UR_Sr, UR_tau = prim_to_cons(rhoR, vR, pR, gamma_rr_f, eos)

        # Package for Riemann solver
        UL_batch = np.stack([UL_D, UL_Sr, UL_tau], axis=1)
        UR_batch = np.stack([UR_D, UR_Sr, UR_tau], axis=1)
        primL_batch = np.stack([rhoL, vL, pL], axis=1)
        primR_batch = np.stack([rhoR, vR, pR], axis=1)

        # Solve Riemann problem to get physical fluxes
        F_phys_batch = riemann_solver.solve_batch(
            UL_batch, UR_batch, primL_batch, primR_batch,
            gamma_rr_f, alpha_f, beta_r_f, eos
        )

        # Multiply by √ĝ α √γ to get fully densitized fluxes
        dens_factor = sqrt_g_hat_f * alpha_f * sqrt_gamma_f
        F_batch = dens_factor[:, None] * F_phys_batch

        return {
            'D': F_batch[:, 0],
            'Sr': F_batch[:, 1],
            'tau': F_batch[:, 2]
        }

    def _get_mesh_spacing(self, grid, r):
        """Get mesh spacing from grid object."""
        if hasattr(grid, 'derivs') and hasattr(grid.derivs, 'dx'):
            return float(grid.derivs.dx)
        elif hasattr(grid, 'dr'):
            return float(grid.dr)
        return float(r[1] - r[0]) if len(r) > 1 else 1.0

    def _apply_ghost_cell_boundaries(self, D, Sr, tau, r):
        """
        Apply boundary conditions to conservative variables.

        Parity mode:
            Inner boundary (r=0): Parity reflection
                - D: even parity
                - Sr: odd parity (radial momentum)
                - tau: even parity
            Outer boundary: Constant extrapolation

        Outflow mode:
            Both boundaries: Constant extrapolation (zero-gradient)
        """
        N = len(r)
        if NUM_GHOSTS > 0:
            if self.boundary_mode == "outflow":
                # Left boundary: outflow (zero-gradient)
                D[:NUM_GHOSTS] = D[NUM_GHOSTS]
                Sr[:NUM_GHOSTS] = Sr[NUM_GHOSTS]
                tau[:NUM_GHOSTS] = tau[NUM_GHOSTS]

                # Right boundary: outflow (zero-gradient)
                last = N - NUM_GHOSTS - 1
                if last >= 0:
                    D[-NUM_GHOSTS:] = D[last]
                    Sr[-NUM_GHOSTS:] = Sr[last]
                    tau[-NUM_GHOSTS:] = tau[last]
            else:
                # Parity mode (default)
                # Inner boundary: parity reflection
                mir = slice(2 * NUM_GHOSTS - 1, NUM_GHOSTS - 1, -1)
                D[:NUM_GHOSTS] = D[mir]       # Even parity
                Sr[:NUM_GHOSTS] = -Sr[mir]    # Odd parity
                tau[:NUM_GHOSTS] = tau[mir]   # Even parity
                
                # Outer boundary: constant extrapolation
                last = N - NUM_GHOSTS - 1
                if last >= 0:
                    D[-NUM_GHOSTS:] = D[last]
                    Sr[-NUM_GHOSTS:] = Sr[last]
                    tau[-NUM_GHOSTS:] = tau[last]

        return D, Sr, tau

    def _apply_rhs_boundary_conditions(self, rhs_D, rhs_Sr, rhs_tau, r):
        """Apply boundary conditions to RHS (same as state variables)."""
        N = len(r)
        if NUM_GHOSTS > 0:
            if self.boundary_mode == "outflow":
                # Left boundary: outflow
                rhs_D[:NUM_GHOSTS] = rhs_D[NUM_GHOSTS]
                rhs_Sr[:NUM_GHOSTS] = rhs_Sr[NUM_GHOSTS]
                rhs_tau[:NUM_GHOSTS] = rhs_tau[NUM_GHOSTS]

                # Right boundary: outflow
                last = N - NUM_GHOSTS - 1
                if last >= 0:
                    rhs_D[-NUM_GHOSTS:] = rhs_D[last]
                    rhs_Sr[-NUM_GHOSTS:] = rhs_Sr[last]
                    rhs_tau[-NUM_GHOSTS:] = rhs_tau[last]
            else:
                # Parity mode
                # Inner boundary
                mir = slice(2 * NUM_GHOSTS - 1, NUM_GHOSTS - 1, -1)
                rhs_D[:NUM_GHOSTS] = rhs_D[mir]
                rhs_Sr[:NUM_GHOSTS] = -rhs_Sr[mir]
                rhs_tau[:NUM_GHOSTS] = rhs_tau[mir]

                # Outer boundary
                last = N - NUM_GHOSTS - 1
                if last >= 0:
                    rhs_D[-NUM_GHOSTS:] = rhs_D[last]
                    rhs_Sr[-NUM_GHOSTS:] = rhs_Sr[last]
                    rhs_tau[-NUM_GHOSTS:] = rhs_tau[last]

        return rhs_D, rhs_Sr, rhs_tau

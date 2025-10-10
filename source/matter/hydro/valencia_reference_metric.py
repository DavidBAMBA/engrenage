# valencia_reference_metric.py
"""
Valencia formulation with reference metric - full 3D BSSN-style implementation.

Follows the exact same tensor algebra pattern as bssnrhs.py:
- Full 3D einsum contractions for all tensor operations
- Spherical symmetry imposed by velocity (v^θ = v^φ = 0) and background metric
- Source terms following NRPy+ GRHD_equations.py structure
- Connection terms with CORRECT signs for covariant divergence

Conservative evolution equations in curved coordinates (two equivalent views):
    Densitized form:  ∂_t(U) + (1/√ĝ) ∂_j[√ĝ F^j] = S
    NRPy form used here:  ∂_t(U) + ∂_j(F̃^j) = S + connection
with F̃^j = α √γ F^j (no √ĝ factor). The explicit connection pieces are
    D, τ:   -Γ̂^k_{kj} F̃^j
    S_i:   -Γ̂^k_{kj} F̃^j_i + Γ̂^l_{ji} F̃^j_l

Where:
    U = (D, S_i, τ)  conserved variables
    F^j = physical fluxes
    S = geometric source terms (K_ij, ∂_iα, ∇̂γ_{ij} couplings)
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
from source.matter.hydro.atmosphere import AtmosphereParams


class ValenciaReferenceMetric:
    """Valencia formulation - full 3D tensor algebra following BSSN pattern."""

    def __init__(self, boundary_mode="parity", *, atmosphere=None,
                 atmosphere_rho=None, p_floor=None, v_max=None):
        """
        Initialize Valencia formulation.

        Parameters
        -----------
        boundary_mode : str
            "parity" - Parity boundary conditions at inner boundary (r=0)
            "outflow" - Outflow (zero-gradient) at both boundaries
        atmosphere : AtmosphereParams, optional
            Centralized atmosphere configuration 
        atmosphere_rho : float, optional
            Deprecated - use atmosphere instead
        p_floor : float, optional
            Deprecated - use atmosphere instead
        v_max : float, optional
            Deprecated - use atmosphere instead
        """
        self.boundary_mode = boundary_mode

        # Handle backward compatibility with old parameter names
        if atmosphere is None:
            # Old-style: individual parameters
            if atmosphere_rho is not None or p_floor is not None or v_max is not None:
                atmosphere = AtmosphereParams(
                    rho_floor=atmosphere_rho if atmosphere_rho is not None else 1e-13,
                    p_floor=p_floor if p_floor is not None else 1e-15,
                    v_max=v_max if v_max is not None else 0.999999
                )
            else:
                # No parameters provided - use defaults
                atmosphere = AtmosphereParams()

        self.atmosphere = atmosphere

    def _extract_geometry(self, r, bssn_vars, spacetime_mode, background):
        """
        Extract geometric quantities from BSSN variables.

        Returns dictionary with all geometric quantities needed for RHS computation.
        """
        N = len(r)
        g = {}

        if spacetime_mode == "fixed_minkowski":
            g['alpha'] = np.ones(N)
            
            # All three shift components are zero
            g['beta_U'] = np.zeros((N, SPACEDIM))
            
            # Identity metric in 3D
            g['gamma_LL'] = np.zeros((N, SPACEDIM, SPACEDIM))
            for i in range(SPACEDIM):
                g['gamma_LL'][:, i, i] = 1.0
            
            # Inverse metric (also identity)
            g['gamma_UU'] = np.copy(g['gamma_LL'])
            
            # Determinant and square root
            g['sqrt_gamma'] = np.ones(N)
            g['sqrt_g_hat_cell'] = background.det_hat_gamma ** 0.5
            g['sqrt_g_hat_face'] = 0.5 * (g['sqrt_g_hat_cell'][:-1] + g['sqrt_g_hat_cell'][1:])
            
        else:
            # Lapse α
            g['alpha'] = np.asarray(bssn_vars.lapse, dtype=float)

            # Shift β^i (all three components)
            g['beta_U'] = np.zeros((N, SPACEDIM))
            if hasattr(bssn_vars, 'shift_U') and bssn_vars.shift_U is not None:
                shift_array = np.asarray(bssn_vars.shift_U)
                if shift_array.ndim >= 2:
                    for i in range(min(SPACEDIM, shift_array.shape[1])):
                        g['beta_U'][:, i] = shift_array[:, i].astype(float)

            # Conformal factor e^{φ}
            phi_arr = np.asarray(bssn_vars.phi, dtype=float)
            e6phi = np.exp(6.0 * phi_arr)
            e4phi = np.exp(4.0 * phi_arr)

            # Physical metric γ_ij = e^{4φ} γ̄_ij (full 3x3 tensor)
            bar_gamma_LL = get_bar_gamma_LL(r, bssn_vars.h_LL, background)
            g['gamma_LL'] = e4phi[:, None, None] * bar_gamma_LL
            
            # Inverse physical metric γ^{ij}
            g['gamma_UU'] = np.linalg.inv(g['gamma_LL'])

            # √γ = e^{6φ} √(det γ̄) (NRPy+ uses e6phi, line 223, 244, 257)
            det_bar_gamma = get_det_bar_gamma(r, bssn_vars.h_LL, background)
            g['sqrt_gamma'] = e6phi * np.sqrt(np.abs(det_bar_gamma) + 1e-30)

            # √ĝ for reference metric
            g['sqrt_g_hat_cell'] = np.sqrt(np.abs(background.det_hat_gamma) + 1e-30)
            g['sqrt_g_hat_face'] = 0.5 * (g['sqrt_g_hat_cell'][:-1] + g['sqrt_g_hat_cell'][1:])

        return g

    def _compute_vU_from_u4U(self, u4U):
        """
        Compute fluid three-velocity from four-velocity: v^i = u^i/u^0.
        
        Args:
            u4U: (N, 4) array with four-velocity components [u^t, u^x, u^y, u^z]
        
        Returns: v^i as (N, 3) array with spatial velocity components
        """
        v_U = np.zeros((len(u4U), SPACEDIM))
        for i in range(SPACEDIM):
            v_U[:, i] = u4U[:, i + 1] / u4U[:, 0]
        
        return v_U

    def _compute_4velocity(self, rho0, v_U, W, g):
        """
        Compute 4-velocity components u^μ following NRPy+ convention.
        
        u^0 = W/α (timelike component, Valencia formulation)
        u^i = W v^i (spatial components)
        
        Args:
            v_U: (N, 3) array with [v^x, v^y, v^z] spatial velocity components
        
        Returns: u^μ as (N, 4) array [u^t, u^x, u^y, u^z]
        """
        N = len(rho0)
        alpha = g['alpha']
        
        # 4-velocity components (NRPy+ line 59-61)
        u4U = np.zeros((N, 4))
        u4U[:, 0] = W / alpha  # u^t = u^0
        
        # All three spatial components
        for i in range(SPACEDIM):
            u4U[:, i + 1] = W * v_U[:, i]  # u^i = W v^i
        
        return u4U

    def _compute_T4UU(self, rho0, v_U, pressure, W, h, g):
        """
        Compute contravariant stress-energy tensor T^{μν} following NRPy+ (line 146-161).
        
        T^{μν} = ρ₀ h u^μ u^ν + P g^{μν}
        
        Args:
            v_U: (N, 3) array with [v^x, v^y, v^z] spatial velocity components
            g: Dictionary with geometric quantities from _extract_geometry
        
        Returns: T^{μν} with components:
            - T^{00} = T^{tt} (energy density in Eulerian frame)
            - T^{0i} = T^{ti} (energy flux / momentum density)
            - T^{ij} (momentum flux / stress tensor)
        """
        N = len(rho0)
        
        # Get 4-velocity
        u4U = self._compute_4velocity(rho0, v_U, W, g)
        
        # Extract geometric quantities (already computed in _extract_geometry)
        alpha = g['alpha']
        beta_U = g['beta_U']  # (N, SPACEDIM)
        gamma_UU = g['gamma_UU']  # (N, SPACEDIM, SPACEDIM)
        
        # Compute g^{μν} components (NRPy+ uses ADM_to_g4UU, line 153)
        # g^{tt} = -1/α²
        g4UU_00 = -1.0 / (alpha ** 2)
        
        # g^{ti} = β^i/α²
        g4UU_0i = beta_U / (alpha[:, None] ** 2)
        
        # g^{ij} = γ^{ij} - β^i β^j / α²
        g4UU_spatial = gamma_UU - np.einsum('xi,xj->xij', beta_U, beta_U) / (alpha[:, None, None] ** 2)
        
        # Compute T^{μν} = ρ₀ h u^μ u^ν + P g^{μν} (NRPy+ line 159-161)
        T4UU = {}
        
        # T^{00} = ρ₀ h u^0 u^0 + P g^{00}
        T4UU['00'] = rho0 * h * u4U[:, 0] * u4U[:, 0] + pressure * g4UU_00
        
        # T^{0i} = ρ₀ h u^0 u^i + P g^{0i} (all three spatial components)
        T4UU['0i'] = np.zeros((N, SPACEDIM))
        for i in range(SPACEDIM):
            T4UU['0i'][:, i] = (
                rho0 * h * u4U[:, 0] * u4U[:, i + 1]
                + pressure * g4UU_0i[:, i]
            )
        
        # T^{ij} = ρ₀ h u^i u^j + P g^{ij} (all nine spatial components)
        T4UU['ij'] = np.zeros((N, SPACEDIM, SPACEDIM))
        for i in range(SPACEDIM):
            for j in range(SPACEDIM):
                T4UU['ij'][:, i, j] = (
                    rho0 * h * u4U[:, i + 1] * u4U[:, j + 1]
                    + pressure * g4UU_spatial[:, i, j]
                )
        
        return T4UU

    def _compute_T4UD(self, T4UU, g):
        """
        Compute mixed stress-energy tensor T^μ_ν = T^{μδ} g_{δν} following NRPy+ (line 176-198).
        
        Needed for momentum flux computation.
        
        Args:
            T4UU: Contravariant stress-energy tensor from _compute_T4UU
            g: Dictionary with geometric quantities from _extract_geometry
        
        Returns: T^μ_ν with all components
        """
        N = len(g['alpha'])
        
        # Extract geometric quantities (already computed in _extract_geometry)
        alpha = g['alpha']
        beta_U = g['beta_U']  # (N, SPACEDIM)
        gamma_LL = g['gamma_LL']  # (N, SPACEDIM, SPACEDIM)
        
        # Build g_{μν} components (NRPy+ uses ADM_to_g4DD)
        # g_{tt} = -α² + β_k β^k (using γ_ij to lower β)
        beta_lower = np.einsum('xij,xj->xi', gamma_LL, beta_U)
        beta_squared = np.einsum('xi,xi->x', beta_U, beta_lower)
        g4DD_00 = -alpha ** 2 + beta_squared
        
        # g_{ti} = β_i (lowered with γ)
        g4DD_0i = beta_lower
        
        # g_{ij} = γ_{ij}
        g4DD_spatial = gamma_LL
        
        # Compute T^μ_ν = T^{μδ} g_{δν} (NRPy+ line 194-198)
        T4UD = {}
        
        # T^0_0 = T^{00} g_{00} + T^{0i} g_{i0}
        T4UD['0_0'] = (
            T4UU['00'] * g4DD_00
            + np.einsum('xi,xi->x', T4UU['0i'], g4DD_0i)
        )
        
        # T^0_j = T^{00} g_{0j} + T^{0i} g_{ij} (all three components)
        T4UD['0_j'] = np.zeros((N, SPACEDIM))
        T4UD['0_j'] = (
            T4UU['00'][:, None] * g4DD_0i
            + np.einsum('xi,xij->xj', T4UU['0i'], g4DD_spatial)
        )
        
        # T^i_j = T^{i0} g_{0j} + T^{ik} g_{kj} (full computation, NRPy+ line 194-198)
        T4UD['i_j'] = np.zeros((N, SPACEDIM, SPACEDIM))
        for i in range(SPACEDIM):
            T4UD['i_j'][:, i, :] = (
                T4UU['0i'][:, i, None] * g4DD_0i
                + np.einsum('xk,xkj->xj', T4UU['ij'][:, i, :], g4DD_spatial)
            )
        
        return T4UD


    def _compute_source_terms(self, rho0, v_U, pressure, W, h, g,
                            bssn_vars, bssn_d1, background, spacetime_mode, r):
        """
        Physical source terms of GRHD equations in Fully 3D implementation.

        Energy source (tau_source_term, NRPy+ line 334-358):
            S_τ = α e^{6φ} √(det γ̄) [K_ij (T^{00} β^i β^j + 2 T^{0i} β^j + T^{ij})
                                    - (T^{00} β^i + T^{0i}) ∂_i α]
        
        Momentum source S_tilde_source_termD:
            S_{S_i} = α e^{6φ} √(det γ̄) [-T^{00} α ∂_i α
                                        + T^0_j (∂_i β^j + Γ̂^j_{ik} β^k)
                                        + (1/2) (T^{tt} β^j β^k + 2 T^{tj} β^k + T^{jk}) ∇̂_i γ_{jk}]
        
        Args:
            v_U: (N, 3) array with [v^x, v^y, v^z] spatial velocity components
            
        Returns:
            src_S_vector: (N, 3) momentum source for all three spatial directions
            src_tau: (N,) energy source
        """
        N = len(r)

        # Handle Minkowski case separately: no explicit hoop-stress sources here.
        # In the NRPy formulation, geometric effects (e.g., 2p/r) arise from connection terms,
        # not explicit sources. So in fixed_minkowski we set sources to zero and rely on
        # the connection block in compute_rhs.
        if spacetime_mode == "fixed_minkowski":
            src_S_vector = np.zeros((N, SPACEDIM))
            src_tau = np.zeros(N)
            return src_S_vector, src_tau

        # Extract geometry
        alpha = g['alpha']
        beta_U = g['beta_U']  # (N, SPACEDIM) - all three components
        
        # Compute e^{6φ} and √(det γ̄) separately for clarity
        phi = np.asarray(bssn_vars.phi, dtype=float)
        e6phi = np.exp(6.0 * phi)
        det_bar_gamma = get_det_bar_gamma(r, bssn_vars.h_LL, background)
        sqrt_det_bar_gamma = np.sqrt(np.abs(det_bar_gamma) + 1e-30)
        
        # This is the full volume element (NRPy+ uses e6phi assuming det γ̄ = 1)
        vol_element = e6phi * sqrt_det_bar_gamma  # = sqrt_gamma in our notation
        
        # Build full 3D metric γ_ij = e^{4φ} γ̄_ij
        e4phi = np.exp(4.0 * phi)
        bar_gamma_LL = get_bar_gamma_LL(r, bssn_vars.h_LL, background)
        gamma_LL = e4phi[:, None, None] * bar_gamma_LL
        
        # Compute stress-energy tensors T^{μν} and T^μ_ν with all three velocity components
        T4UU = self._compute_T4UU(rho0, v_U, pressure, W, h, g)
        T4UD = self._compute_T4UD(T4UU, g)
        
        # ====================================================================
        # ENERGY SOURCE TERM (NRPy+ line 334-358)
        # ====================================================================
        
        # Extrinsic curvature K_ij = e^{4φ} Ā_ij + (K/3) γ_ij
        K = np.asarray(bssn_vars.K, dtype=float)
        bar_A_LL = get_bar_A_LL(r, bssn_vars, background)
        K_LL = e4phi[:, None, None] * bar_A_LL + (K / 3.0)[:, None, None] * gamma_LL
        
        # Lapse derivatives ∂_i α (all three spatial directions)
        dalpha_dx = np.zeros((N, SPACEDIM))
        if hasattr(bssn_d1, 'lapse') and bssn_d1.lapse is not None:
            dalpha_dx = np.asarray(bssn_d1.lapse)
        
        # Term 1: K_ij contraction (NRPy+ line 343-351)
        # Tensor block: T^{00} β^i β^j + 2 T^{0i} β^j + T^{ij}
        tensor_block = (
            T4UU['00'][:, None, None] * np.einsum('xi,xj->xij', beta_U, beta_U)
            + 2.0 * np.einsum('xi,xj->xij', T4UU['0i'], beta_U)
            + T4UU['ij']
        )
        
        term1_tau = np.einsum('xij,xij->x', K_LL, tensor_block)
        
        # Term 2: lapse derivative term (NRPy+ line 352-357)
        # -(T^{00} β^i + T^{0i}) ∂_i α
        term2_tau = -(
            np.einsum('x,xi,xi->x', T4UU['00'], beta_U, dalpha_dx)
            + np.einsum('xi,xi->x', T4UU['0i'], dalpha_dx)
        )
        
        # Combine with volume element (NRPy+ line 358)
        src_tau = alpha * vol_element * (term1_tau + term2_tau)
        
        # ====================================================================
        # MOMENTUM SOURCE TERM (NRPy+ line 388-438)
        # ====================================================================

        # Shift derivatives ∂_i β^j (all spatial components)
        dbeta_dx = np.zeros((N, SPACEDIM, SPACEDIM))
        if hasattr(bssn_d1, 'shift_U') and bssn_d1.shift_U is not None:
            shift_d1 = np.asarray(bssn_d1.shift_U)
            if shift_d1.ndim >= 3:
                dbeta_dx = shift_d1.copy()

        # Reference metric Christoffel symbols Γ̂^i_{jk}
        hat_chris = background.hat_christoffel

        # Covariant derivative of shift: ∇̂_i β^j = ∂_i β^j + Γ̂^j_{ik} β^k
        # (NRPy+ line 422-423)
        hatD_beta_U = dbeta_dx + np.einsum('xjik,xk->xij', hat_chris, beta_U)

        # Covariant derivative of metric: ∇̂_i γ_{jk} (NRPy+ line 407-415)
        dphi_dx = np.zeros((N, SPACEDIM))
        if hasattr(bssn_d1, 'phi') and bssn_d1.phi is not None:
            dphi_dx = np.asarray(bssn_d1.phi)

        hat_D_bar_gamma = get_hat_D_bar_gamma_LL(r, bssn_vars.h_LL, bssn_d1.h_LL, background)

        # ∇̂_i γ_{jk} = e^{4φ} [4 γ̄_{jk} ∂_i φ + ∂_i γ̄_{jk} - Γ̂^l_{ij} γ̄_{lk} - Γ̂^l_{ik} γ̄_{jl}]
        hatD_gamma_LL = np.zeros((N, SPACEDIM, SPACEDIM, SPACEDIM))
        for i in range(SPACEDIM):
            for j in range(SPACEDIM):
                for k in range(SPACEDIM):
                    # Start with partial derivatives and phi term (NRPy+ line 410-411)
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

        # Term 1: -T^{00} α ∂_i α (NRPy+ line 418)
        # All three spatial components
        first_term = -T4UU['00'][:, None] * alpha[:, None] * dalpha_dx

        # Term 2: T^0_j ∇̂_i β^j (NRPy+ line 420-423)
        # All three spatial components
        second_term = np.einsum('xj,xij->xi', T4UD['0_j'], hatD_beta_U)

        # Term 3: (1/2) (T^{tt} β^j β^k + 2 T^{tj} β^k + T^{jk}) ∇̂_i γ_{jk}
        # This is the SAME tensor_block as in the energy equation (NRPy+ line 425-433)
        # NOT just T^{jk}, but the full combination with shift terms
        tensor_block_momentum = (
            T4UU['00'][:, None, None] * np.einsum('xj,xk->xjk', beta_U, beta_U)
            + 2.0 * np.einsum('xj,xk->xjk', T4UU['0i'], beta_U)
            + T4UU['ij']
        )
        third_term = 0.5 * np.einsum('xjk,xijk->xi', tensor_block_momentum, hatD_gamma_LL)

        # Combine with volume element (NRPy+ line 436-438)
        # This gives all three spatial components of the momentum source
        src_S_vector = alpha[:, None] * vol_element[:, None] * (
            first_term + second_term + third_term
        )
        
        # Return full 3D momentum source vector and energy source
        return src_S_vector, src_tau  # (N, 3), (N,)

    def compute_rhs(self, D, S_tildeD, tau, rho0, v_U, pressure, W, h,
                    r, bssn_vars, bssn_d1, background, spacetime_mode,
                    eos, grid, reconstructor, riemann_solver):
        """
        Compute RHS of Valencia equations - fully 3D implementation.

        Variables:
            D: (N,) - densitized rest-mass density
        S_tildeD: (N,) or (N, 3) - densitized momentum (1D or 3D format)
        tau: (N,) - densitized energy
        v_U: (N,) or (N, 3) - spatial velocity (1D or 3D format)
        W: (N,) or None - Lorentz factor (computed if None)
        h: (N,) or None - specific enthalpy (computed if None)
    
        Returns:
        rhs_D: (N,) - time derivative of D
        rhs_S_tildeD: (N, 3) - time derivative of momentum vector
        rhs_tau: (N,) - time derivative of tau
        """
        
        dr = self._get_mesh_spacing(grid, r)
        N = len(r)

        # Track original momentum dimensionality to preserve caller-facing API
        _S_input = np.asarray(S_tildeD)
        _return_radial_only = (_S_input.ndim == 1) or (_S_input.ndim == 2 and _S_input.shape[1] == 1)

        # Coerce inputs to 3D vector form where appropriate (general 3D API)
        # v_U: allow 1D (radial) input; expand to (N,3) with zeros in angular comps
        if v_U is None:
            v_U = np.zeros((N, SPACEDIM))
        else:
            v_arr = np.asarray(v_U)
            if v_arr.ndim == 1:
                v_U = np.zeros((N, SPACEDIM), dtype=float)
                v_U[:, 0] = v_arr
            elif v_arr.ndim == 2 and v_arr.shape[1] != SPACEDIM:
                # Treat as radial-only if second dim mismatched
                v_U = np.zeros((N, SPACEDIM), dtype=float)
                v_U[:, 0] = v_arr[:, 0]
            else:
                v_U = v_arr

        # S_tildeD: allow 1D (radial) input; expand to (N,3)
        S_arr = np.asarray(S_tildeD)
        if S_arr.ndim == 1:
            S_new = np.zeros((N, SPACEDIM), dtype=float)
            S_new[:, 0] = S_arr
            S_tildeD = S_new
        elif S_arr.ndim == 2 and S_arr.shape[1] != SPACEDIM:
            S_new = np.zeros((N, SPACEDIM), dtype=float)
            S_new[:, 0] = S_arr[:, 0]
            S_tildeD = S_new
        else:
            S_tildeD = S_arr

            # Convert velocity to (N, 3) if needed
        v_U = np.asarray(v_U)
        if v_U.ndim == 1:
            v_U_3d = np.zeros((N, SPACEDIM))
            v_U_3d[:, 0] = v_U
            v_U = v_U_3d
        elif v_U.shape[1] != SPACEDIM:
            raise ValueError(f"Velocity shape {v_U.shape} incompatible with SPACEDIM={SPACEDIM}")
        

        # Extract geometry (α, β^i, γ_ij, √γ, √ĝ, etc.) early, as W may need γ_ij
        g = self._extract_geometry(r, bssn_vars, spacetime_mode, background)

        # Compute W and h from primitives if not provided
        if W is None:
            # v² = γ_ij v^i v^j
            gamma_LL = g['gamma_LL']  # (N, 3, 3)
            v_squared = np.einsum('xij,xi,xj->x', gamma_LL, v_U, v_U)
            # Lorentz factor: W = 1/√(1 - v²)
            W = 1.0 / np.sqrt(np.maximum(1.0 - v_squared, 1e-16))

        if h is None:
            eps = eos.eps_from_rho_p(rho0, pressure)
            h = 1.0 + eps + pressure / np.maximum(rho0, 1e-30)

        # Copy and apply boundary conditions to conservatives
        D = D.copy()
        S_tildeD = S_tildeD.copy()
        tau = tau.copy()
        D, S_tildeD, tau = self._apply_ghost_cell_boundaries(D, S_tildeD, tau, r)

        # Compute NRPy-style partial fluxes at interfaces: α_face √γ_face F_phys
        flux_hat = self._compute_interface_fluxes(
            rho0, v_U, pressure, g, r, eos, reconstructor, riemann_solver
        )

        # Reference metric Christoffel symbols Γ̂^i_{jk}
        hat_chris = background.hat_christoffel  # (N, 3, 3, 3)

        # Partial fluxes at faces (no √ĝ)
        F_D_face = flux_hat['D']
        F_S_tildeD_face = flux_hat['S_tildeD']  # Now (N_faces, 3) for all three components
        F_tau_face = flux_hat['tau']

        # ====================================================================
        # FLUX DIVERGENCE (NRPy form): -∂_r(F̃^r)
        #   with F̃^r = α √γ F_phys^r (no √ĝ factor here)
        # ====================================================================

        rhs_D = np.zeros(N)
        rhs_S_tildeD = np.zeros((N, SPACEDIM))  # Full 3D momentum RHS
        rhs_tau = np.zeros(N)

        # Compute divergence in interior cells
        for i in range(NUM_GHOSTS, N - NUM_GHOSTS):
            # NRPy-style partial derivative form: divide only by Δr
            inv_vol = 1.0 / (dr + 1e-30)

            rhs_D[i] = -(F_D_face[i] - F_D_face[i-1]) * inv_vol
            
            # All three momentum components
            rhs_S_tildeD[i, :] = -(F_S_tildeD_face[i, :] - F_S_tildeD_face[i-1, :]) * inv_vol
            
            rhs_tau[i] = -(F_tau_face[i] - F_tau_face[i-1]) * inv_vol

        # ====================================================================
        # CONNECTION TERMS (NRPy form):
        #   D, τ:      -Γ̂^k_{kr} F̃^r
        #   S_i:       -Γ̂^k_{kr} F̃^r_i + Γ̂^l_{ri} F̃^r_l
        # ====================================================================
        
        # Build sqrt_gamma consistently: use hat determinant in Minkowski, bar determinant otherwise.
        alpha = g['alpha']
        if spacetime_mode == "fixed_minkowski":
            sqrt_gamma = g['sqrt_gamma']
        else:
            phi = np.asarray(bssn_vars.phi, dtype=float)
            e6phi = np.exp(6.0 * phi)
            det_bar_gamma = get_det_bar_gamma(r, bssn_vars.h_LL, background)
            sqrt_det_bar_gamma = np.sqrt(np.abs(det_bar_gamma) + 1e-30)
            sqrt_gamma = e6phi * sqrt_det_bar_gamma

        # Compute 4-velocity and stress-energy
        u4U = self._compute_4velocity(rho0, v_U, W, g)
        T4UU = self._compute_T4UU(rho0, v_U, pressure, W, h, g)

        # Conservative density
        rho_star = alpha * sqrt_gamma * rho0 * u4U[:, 0]

        # Physical fluxes at cell centers
        fD_U = np.zeros((N, SPACEDIM))
        for j in range(SPACEDIM):
            fD_U[:, j] = rho_star * v_U[:, j]

        fTau_U = np.zeros((N, SPACEDIM))
        for j in range(SPACEDIM):
            fTau_U[:, j] = alpha ** 2 * sqrt_gamma * T4UU['0i'][:, j] - rho_star * v_U[:, j]

        # Momentum partial flux tensor F̃^j_i = α √γ T^j_i
        T4UD = self._compute_T4UD(T4UU, g)
        F_S_no_ghat = np.zeros((N, SPACEDIM, SPACEDIM))
        for j in range(SPACEDIM):
            for i in range(SPACEDIM):
                F_S_no_ghat[:, j, i] = alpha * sqrt_gamma * T4UD['i_j'][:, j, i]

        # Connection contributions
        Gamma_trace = np.einsum('xkkj->xj', hat_chris)
        conn_D = -np.einsum('xj,xj->x', Gamma_trace, fD_U)
        conn_tau = -np.einsum('xj,xj->x', Gamma_trace, fTau_U)
        conn_S_tildeD = (
            -np.einsum('xl,xli->xi', Gamma_trace, F_S_no_ghat)
            + np.einsum('xlji,xjl->xi', hat_chris, F_S_no_ghat)
        )

        # Add connection terms to RHS
        rhs_D += conn_D
        rhs_tau += conn_tau
        rhs_S_tildeD += conn_S_tildeD  # Now adds to all three components

        # ====================================================================
        # SOURCE TERMS (geometric couplings - NRPy+ style)
        # ====================================================================
        src_S_tildeD, src_tau = self._compute_source_terms(
            rho0, v_U, pressure, W, h, g, bssn_vars, bssn_d1,
            background, spacetime_mode, r
        )  # Returns (N,3), (N,)

        rhs_S_tildeD += src_S_tildeD  # Vector addition - all three components
        rhs_tau += src_tau

        # Preserve backward-compatible shape: if input momentum was 1D, return RHS radial as (N,)
        if _return_radial_only:
            rhs_S_out = rhs_S_tildeD[:, 0]
        else:
            rhs_S_out = rhs_S_tildeD

        return rhs_D, rhs_S_out, rhs_tau  # (N,), (N,) or (N,3), (N,)

    def _compute_interface_fluxes(self, rho0, v_U, pressure, g, r,
                                eos, reconstructor, riemann_solver):
        """
        Compute NRPy-style partial fluxes at cell interfaces: F̃_face = α_face √γ_face F_phys.

        GENERAL 3D STRUCTURE, 1D EVOLUTION:
        ====================================
        - Velocity v_U has shape (N, 3) with all spatial components
        - Only first spatial direction (index 0) fluxes computed via Riemann solver
        - Other spatial directions (indices 1,2) have zero flux

        Coordinate interpretation depends on background:
            - Spherical: v_U = [v^r, v^θ, v^φ], evolves v^r flux only

        This is a 1D solver limitation, NOT a physics assumption.
        All tensor algebra (T^μν, sources, connections) remains fully 3D.

        Args:
            v_U: (N, 3) contravariant velocity components in coordinate basis
            r: Coordinate for reconstruction (interpretation depends on background)

        Returns:
            Dictionary with partial-flux arrays at interfaces (no √ĝ factor):
                'D': (N_faces,) density flux
                'S_tildeD': (N_faces, 3) momentum flux (only [:, 0] non-zero)
                'tau': (N_faces,) energy flux
        """
        N = len(r)
        
        # Extract first spatial component for 1D reconstruction
        # This is v^r (spherical), v^ρ (cylindrical), or v^x (Cartesian)
        # depending on the background coordinate system
        v_primary = v_U[:, 0]
        
        # Reconstruct primitives to left/right states at interfaces
        # Map Valencia boundary_mode to reconstructor boundary_type
        recon_boundary = "reflecting" if self.boundary_mode == "parity" else "outflow"
        (rhoL, vL, pL), (rhoR, vR, pR) = reconstructor.reconstruct_primitive_variables(
            rho0, v_primary, pressure, x=r, boundary_type=recon_boundary
        )
        
        # Extract interior faces (exclude ghost zones)
        rhoL, vL, pL = rhoL[1:-1], vL[1:-1], pL[1:-1]
        rhoR, vR, pR = rhoR[1:-1], vR[1:-1], pR[1:-1]
        
        # Interpolate geometry to faces
        alpha_f = 0.5 * (g['alpha'][:-1] + g['alpha'][1:])
        sqrt_gamma_f = 0.5 * (g['sqrt_gamma'][:-1] + g['sqrt_gamma'][1:])
        sqrt_g_hat_f = g['sqrt_g_hat_face']
        
        # Interpolate shift and metric to faces
        beta_U_f = 0.5 * (g['beta_U'][:-1] + g['beta_U'][1:])  # (N-1, 3)
        gamma_LL_f = 0.5 * (g['gamma_LL'][:-1] + g['gamma_LL'][1:])  # (N-1, 3, 3)
        
        # For 1D Riemann solver, extract first spatial components
        # (direction depends on coordinate system from background)
        beta_primary_f = beta_U_f[:, 0]  # β^r or β^ρ or β^x
        gamma_primary_primary_f = gamma_LL_f[:, 0, 0]  # γ_rr or γ_ρρ or γ_xx
        
        # Enforce reflecting condition at first physical interior face (r≈0)
        # Only applies to spherical-like coordinates with reflecting boundary
        if recon_boundary == "reflecting" and (len(vL) > NUM_GHOSTS):
            k0 = NUM_GHOSTS  # interface between cells NUM_GHOSTS and NUM_GHOSTS+1
            if k0 < len(vL):
                vL[k0] = 0.0
                vR[k0] = 0.0
                rho_ref = rho0[NUM_GHOSTS]
                p_ref = pressure[NUM_GHOSTS]
                rhoL[k0] = rho_ref
                rhoR[k0] = rho_ref
                pL[k0] = p_ref
                pR[k0] = p_ref
        
        # Atmosphere flattening: near-vacuum regions
        dens_floor = max(self.atmosphere.rho_floor, 1e-20)
        thr = 30.0 * dens_floor
        if len(rhoL) > 0:
            mask = (rhoL < thr) | (rhoR < thr)
            if np.any(mask):
                rho_avg = 0.5 * (rhoL + rhoR)
                p_avg = np.maximum(0.5 * (pL + pR), self.atmosphere.p_floor)
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
                atmosphere=self.atmosphere,
                gamma_rr=gamma_primary_primary_f  # More general name
            )
        
        # Convert primitives to conservatives at interfaces
        # prim_to_cons computes: D, S_primary, τ where S_primary is momentum
        # in the primary (first) spatial direction
        UL_D, UL_S_primary, UL_tau = prim_to_cons(
            rhoL, vL, pL, gamma_primary_primary_f, eos
        )
        UR_D, UR_S_primary, UR_tau = prim_to_cons(
            rhoR, vR, pR, gamma_primary_primary_f, eos
        )
        
        # Package for Riemann solver
        UL_batch = np.stack([UL_D, UL_S_primary, UL_tau], axis=1)
        UR_batch = np.stack([UR_D, UR_S_primary, UR_tau], axis=1)
        primL_batch = np.stack([rhoL, vL, pL], axis=1)
        primR_batch = np.stack([rhoR, vR, pR], axis=1)
        
        # Solve Riemann problem to get physical fluxes
        # Returns: [F_D, F_S_primary, F_tau] in primary spatial direction
        F_phys_batch = riemann_solver.solve_batch(
            UL_batch, UR_batch, primL_batch, primR_batch,
            gamma_primary_primary_f, alpha_f, beta_primary_f, eos
        )
        
        # Multiply by α √γ to get NRPy-style partial fluxes (no √ĝ here)
        dens_factor = alpha_f * sqrt_gamma_f
        F_batch = dens_factor[:, None] * F_phys_batch
        
        # Construct 3D momentum flux array
        # In 1D evolution: only primary direction has flux, others are zero
        # This is correct because:
        # - No transverse Riemann problems solved
        # - Transverse momenta evolve via connection/source terms only
        N_faces = len(F_batch)
        F_S_tildeD_face = np.zeros((N_faces, SPACEDIM))
        F_S_tildeD_face[:, 0] = F_batch[:, 1]  # Primary direction flux
        # F_S_tildeD_face[:, 1] = 0  # Secondary direction (no flux in 1D)
        # F_S_tildeD_face[:, 2] = 0  # Tertiary direction (no flux in 1D)
        
        return {
            'D': F_batch[:, 0],           # Density flux
            'S_tildeD': F_S_tildeD_face,  # (N_faces, 3) momentum flux vector
            'tau': F_batch[:, 2]          # Energy flux
        }

    def _get_mesh_spacing(self, grid, r):
        """
        Get mesh spacing from grid object.
        
        For 3D grids, this returns the radial spacing.
        Full 3D implementation would need dx, dy, dz or dr, dtheta, dphi.
        """
        if hasattr(grid, 'derivs') and hasattr(grid.derivs, 'dx'):
            return float(grid.derivs.dx)
        elif hasattr(grid, 'dr'):
            return float(grid.dr)
        elif hasattr(grid, 'spacing') and np.isscalar(grid.spacing):
            return float(grid.spacing)
        elif hasattr(grid, 'spacing') and hasattr(grid.spacing, '__len__'):
            # For 3D grids with multiple spacings, return radial spacing
            return float(grid.spacing[0])
        else:
            # Fallback: compute from radial coordinate array
            return float(r[1] - r[0]) if len(r) > 1 else 1.0

    def _apply_ghost_cell_boundaries(self, D, S_tildeD, tau, r):
        """
        Apply boundary conditions to conservative variables.
        
        Full 3D implementation handling all momentum components.
        
        Args:
            D: (N,) density array
            S_tildeD: (N, 3) momentum array with all three spatial components
            tau: (N,) energy array
            r: (N,) radial coordinate array
            
        Parity mode:
            Inner boundary (r=0): Parity reflection
                - D: even parity
                - S_r: odd parity (radial momentum)
                - S_θ, S_φ: even parity (angular momenta)
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
                for i in range(SPACEDIM):
                    S_tildeD[:NUM_GHOSTS, i] = S_tildeD[NUM_GHOSTS, i]
                tau[:NUM_GHOSTS] = tau[NUM_GHOSTS]
                
                # Right boundary: outflow (zero-gradient)
                last = N - NUM_GHOSTS - 1
                if last >= 0:
                    D[-NUM_GHOSTS:] = D[last]
                    for i in range(SPACEDIM):
                        S_tildeD[-NUM_GHOSTS:, i] = S_tildeD[last, i]
                    tau[-NUM_GHOSTS:] = tau[last]
                    
            else:
                # Parity mode (default)
                # Inner boundary: parity reflection at r=0
                mir = slice(2 * NUM_GHOSTS - 1, NUM_GHOSTS - 1, -1)
                
                # Density: even parity
                D[:NUM_GHOSTS] = D[mir]
                
                # Momentum components with appropriate parity
                # S_r (radial): odd parity (flips sign across r=0)
                S_tildeD[:NUM_GHOSTS, 0] = -S_tildeD[mir, 0]
                
                # S_θ, S_φ (angular): even parity (no sign flip)
                # In spherical symmetry these should be zero anyway
                for i in range(1, SPACEDIM):
                    S_tildeD[:NUM_GHOSTS, i] = S_tildeD[mir, i]
                
                # Energy: even parity
                tau[:NUM_GHOSTS] = tau[mir]
                
                # Outer boundary: constant extrapolation
                last = N - NUM_GHOSTS - 1
                if last >= 0:
                    D[-NUM_GHOSTS:] = D[last]
                    for i in range(SPACEDIM):
                        S_tildeD[-NUM_GHOSTS:, i] = S_tildeD[last, i]
                    tau[-NUM_GHOSTS:] = tau[last]
        
        return D, S_tildeD, tau

    def _apply_rhs_boundary_conditions(self, rhs_D, rhs_S_tildeD, rhs_tau, r):
        """
        Apply boundary conditions to RHS arrays.
        
        Full 3D implementation handling all momentum components.
        Same parity rules as state variables.
        
        Args:
            rhs_D: (N,) density RHS
            rhs_S_tildeD: (N, 3) momentum RHS with all three spatial components
            rhs_tau: (N,) energy RHS
            r: (N,) radial coordinate array
        """
        N = len(r)
        
        if NUM_GHOSTS > 0:
            if self.boundary_mode == "outflow":
                # Left boundary: outflow
                rhs_D[:NUM_GHOSTS] = rhs_D[NUM_GHOSTS]
                for i in range(SPACEDIM):
                    rhs_S_tildeD[:NUM_GHOSTS, i] = rhs_S_tildeD[NUM_GHOSTS, i]
                rhs_tau[:NUM_GHOSTS] = rhs_tau[NUM_GHOSTS]
                
                # Right boundary: outflow
                last = N - NUM_GHOSTS - 1
                if last >= 0:
                    rhs_D[-NUM_GHOSTS:] = rhs_D[last]
                    for i in range(SPACEDIM):
                        rhs_S_tildeD[-NUM_GHOSTS:, i] = rhs_S_tildeD[last, i]
                    rhs_tau[-NUM_GHOSTS:] = rhs_tau[last]
                    
            else:
                # Parity mode
                # Inner boundary
                mir = slice(2 * NUM_GHOSTS - 1, NUM_GHOSTS - 1, -1)
                
                # Density RHS: even parity
                rhs_D[:NUM_GHOSTS] = rhs_D[mir]
                
                # Momentum RHS components with appropriate parity
                # Radial component: odd parity
                rhs_S_tildeD[:NUM_GHOSTS, 0] = -rhs_S_tildeD[mir, 0]
                
                # Angular components: even parity
                for i in range(1, SPACEDIM):
                    rhs_S_tildeD[:NUM_GHOSTS, i] = rhs_S_tildeD[mir, i]
                
                # Energy RHS: even parity
                rhs_tau[:NUM_GHOSTS] = rhs_tau[mir]
                
                # Outer boundary
                last = N - NUM_GHOSTS - 1
                if last >= 0:
                    rhs_D[-NUM_GHOSTS:] = rhs_D[last]
                    for i in range(SPACEDIM):
                        rhs_S_tildeD[-NUM_GHOSTS:, i] = rhs_S_tildeD[last, i]
                    rhs_tau[-NUM_GHOSTS:] = rhs_tau[last]
        
        return rhs_D, rhs_S_tildeD, rhs_tau
    

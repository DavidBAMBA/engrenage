# valencia_reference_metric.py
"""Valencia formulation with reference metric - full 3D."""

import numpy as np

from source.bssn.tensoralgebra import (
    SPACEDIM,
    get_bar_gamma_LL,
    get_bar_A_LL,
    get_hat_D_bar_gamma_LL
)
from source.core.spacing import NUM_GHOSTS
from source.matter.hydro.cons2prim import prim_to_cons
from source.matter.hydro.atmosphere import AtmosphereParams
from source.matter.hydro.geometry import (
    GeometryState,
    compute_4velocity,
    compute_g4DD,
    compute_g4UU
    )


class ValenciaReferenceMetric:
    """Valencia formulation - full 3D tensor algebra following BSSN pattern."""

    def __init__(self, boundary_mode="parity", *, atmosphere):
        """
        Initialize Valencia formulation.

        Parameters
        -----------
        boundary_mode : str
            "parity" - Parity boundary conditions at inner boundary (r=0)
            "outflow" - Outflow (zero-gradient) at both boundaries
        atmosphere : AtmosphereParams
            Atmosphere configuration (required)
        """
        self.boundary_mode = boundary_mode
        self.atmosphere = atmosphere

        # Geometric quantities
        # These will be computed by _extract_geometry()
        self.alpha = None              # Lapse function
        self.beta_U = None             # Shift vector (N, 3)
        self.gamma_LL = None           # Physical metric (N, 3, 3)
        self.gamma_UU = None           # Inverse metric (N, 3, 3)
        self.e6phi = None              # e^(6φ) conformal factor
        self.sqrt_g_hat_cell = None    # √ĝ at cell centers
        self.sqrt_g_hat_face = None    # √ĝ at cell faces
        self.dx = None                 # Mesh spacing (computational coordinate)

        # GeometryState container (bundles 1D components for passing to other modules)
        self._geom = None

    def _extract_geometry(self, r, bssn_vars, spacetime_mode, background, grid):
        """
        Extract geometric quantities from BSSN variables and store as class attributes.

        IMPORTANT RESCALING NOTES:
        - bssn_vars.shift_U is the RESCALED shift (needs inverse_scaling_vector)
        - bssn_vars.h_LL is the RESCALED deviation (already includes scaling_matrix)
        - bar_gamma_LL = h_LL * scaling_matrix + hat_gamma_LL (includes scale factors)
        - Physical shift: beta^i = inverse_scaling_vector^i * shift_U^i

        Stores all geometric quantities as class attributes, including mesh spacing.
        """
        # Mesh spacing in computational coordinate (uniform)
        self.dx = float(grid.derivs.dx)

        # Lapse alpha
        self.alpha = np.asarray(bssn_vars.lapse, dtype=float)

        # Shift β^i (all three components)
        self.beta_U = np.asarray(bssn_vars.shift_U, dtype=float) * background.inverse_scaling_vector

        # Conformal factor e^{φ}
        phi_arr = np.asarray(bssn_vars.phi, dtype=float)
        e6phi = np.exp(6.0 * phi_arr)
        e4phi = np.exp(4.0 * phi_arr)

        # Physical metric γ_ij = e^{4φ} γ̄_ij (full 3x3 tensor)
        bar_gamma_LL = get_bar_gamma_LL(r, bssn_vars.h_LL, background)
        self.gamma_LL = e4phi[:, None, None] * bar_gamma_LL

        # Inverse physical metric γ^{ij}
        self.gamma_UU = np.linalg.inv(self.gamma_LL)

        self.e6phi = e6phi

        # √ĝ for reference metric
        self.sqrt_g_hat_cell = np.sqrt(np.abs(background.det_hat_gamma) + 1e-30)
        self.sqrt_g_hat_face = 0.5 * (self.sqrt_g_hat_cell[:-1] + self.sqrt_g_hat_cell[1:])

        # Create GeometryState container with 1D components (for cons2prim, riemann, etc.)
        self._geom = GeometryState(
            alpha=self.alpha,
            beta_r=self.beta_U[:, 0],      # Radial component of shift
            gamma_rr=self.gamma_LL[:, 0, 0],  # Radial component of metric
            e6phi=self.e6phi,
            beta_U=self.beta_U,
            gamma_LL=self.gamma_LL,
            gamma_UU=self.gamma_UU
        )

    def _compute_T4UU(self, rho0, v_U, pressure, W, h):
        """
        Compute contravariant stress-energy tensor T^{μν}

        T^{μν} = ρ₀ h u^μ u^ν + P g^{μν}
        Returns:
            tuple: (T00, T0i, Tij) - Components of contravariant stress-energy tensor T^{μν}
                - T00: (N,) array - T^{00} = T^{tt} (energy density in Eulerian frame)
                - T0i: (N, 3) array - T^{0i} = T^{ti} (energy flux / momentum density)
                - Tij: (N, 3, 3) array - T^{ij} (momentum flux / stress tensor)
        """
        # Get 4-velocity (using geometry module)
        u4U = compute_4velocity(v_U, W, self.alpha, self.beta_U)

        # Get contravariant 4-metric g^{μν} (using geometry module)
        g4UU = compute_g4UU(self.alpha, self.beta_U, self.gamma_UU)

        # Compute T^{μν} = ρ₀ h u^μ u^ν + P g^{μν}
        u0U = u4U[:, 0]  # (N,)
        uiU = u4U[:, 1:]  # (N, 3)
        rho_h = rho0 * h  # (N,)

        # T^{00} = ρ₀ h u^0 u^0 + P g^{00}
        T00 = rho_h * u0U * u0U + pressure * g4UU[:, 0, 0]

        # T^{0i} = ρ₀ h u^0 u^i + P g^{0i}
        T0i = (rho_h * u0U)[:, None] * uiU + pressure[:, None] * g4UU[:, 0, 1:]

        # T^{ij} = ρ₀ h u^i u^j + P g^{ij}
        Tij = rho_h[:, None, None] * np.einsum('xi,xj->xij', uiU, uiU) + pressure[:, None, None] * g4UU[:, 1:, 1:]

        return T00, T0i, Tij

    def _compute_T4UD(self, T00, T0i, Tij):
        """
        Compute mixed stress-energy tensor T^μ_ν = T^{μδ} g_{δν}.

        Returns:
            tuple: (T0_0, T0_j, Ti_j) - Components of mixed stress-energy tensor T^μ_ν
                - T0_0: (N,) array - T^0_0
                - T0_j: (N, 3) array - T^0_j
                - Ti_j: (N, 3, 3) array - T^i_j
        """
        # Get covariant 4-metric g_{μν} (using geometry module)
        g4DD = compute_g4DD(self.alpha, self.beta_U, self.gamma_LL)

        # Extract components for readability
        g4DD_00 = g4DD[:, 0, 0]
        g4DD_0i = g4DD[:, 0, 1:]  # (N, 3)
        g4DD_spatial = g4DD[:, 1:, 1:]  # (N, 3, 3)

        # Compute T^μ_ν = T^{μδ} g_{δν}

        # T^0_0 = T^{00} g_{00} + T^{0i} g_{i0}
        T0_0 = T00 * g4DD_00 + np.einsum('xi,xi->x', T0i, g4DD_0i)


        # T^0_j = T^{00} g_{0j} + T^{0i} g_{ij}
        T0_j = T00[:, None] * g4DD_0i + np.einsum('xi,xij->xj', T0i, g4DD_spatial)

        # T^i_j = T^{i0} g_{0j} + T^{ik} g_{kj}
        Ti_j = T0i[:, :, None] * g4DD_0i[:, None, :] + np.einsum('xik,xkj->xij', Tij, g4DD_spatial)

        return T0_0, T0_j, Ti_j


#------------------------------------------------------------------------------
# GRHD source, flux, and connection term computations
#------------------------------------------------------------------------------
    def _compute_source_terms(self, rho0, v_U, pressure, W, h,
                            bssn_vars, bssn_d1, background, spacetime_mode, r):
        """
        Physical source terms of GRHD equations in Fully 3D implementation.

        Energy source (tau_source_term,  line 334-358):
            S_τ = alpha e^{6φ}  [  K_ij (T^{00} β^i β^j + 2 T^{0i} β^j + T^{ij})
                                - (T^{00} β^i + T^{0i}) ∂_i alpha]

        Momentum source S_source_termD:
            S_{S_i} = alpha e^{6φ} [ -T^{00} alpha ∂_i alpha
                                 + T^0_j (∂_i β^j + Γ̂^j_{ik} β^k)
                                 + (1/2) (T^{00} β^j β^k + 2 T^{0j} β^k + T^{jk}) ∇̂^_i γ_{jk}]

        Returns:
            src_S_vector: (N, 3) momentum source for all three spatial directions
            src_tau: (N,) energy source
        """
        N = len(r)

        # Handle Minkowski case separately (are zero)
        if spacetime_mode == "fixed_minkowski":
            src_S_vector = np.zeros((N, SPACEDIM))
            src_tau = np.zeros(N)
            return src_S_vector, src_tau

        # Geometric quantities from class attributes
        alpha = self.alpha
        beta_U = self.beta_U  # (N, SPACEDIM) - all three components

        # Compute e^{6φ}
        phi = np.asarray(bssn_vars.phi, dtype=float)
        e6phi = np.exp(6.0 * phi)

        e4phi = np.exp(4.0 * phi)
        bar_gamma_LL = get_bar_gamma_LL(r, bssn_vars.h_LL, background)
        gamma_LL = e4phi[:, None, None] * bar_gamma_LL

        # Compute stress-energy tensors T^{μν} and T^μ_ν with all three velocity components
        TUU_00, TUU_0i, TUU_ij = self._compute_T4UU(rho0, v_U, pressure, W, h)
        _, TUD_0j, _ = self._compute_T4UD(TUU_00, TUU_0i, TUU_ij)

        # ====================================================================
        # ENERGY SOURCE TERM ( line 334-358)
        # ====================================================================

        # Extrinsic curvature K_ij = e^{4φ} Ā_ij + (K/3) γ_ij
        K = np.asarray(bssn_vars.K, dtype=float)
        bar_A_LL = get_bar_A_LL(r, bssn_vars, background)
        K_LL = e4phi[:, None, None] * bar_A_LL + (K / 3.0)[:, None, None] * gamma_LL

        # Lapse derivatives ∂_i alpha (all three spatial directions)
        dalpha_dx = np.asarray(bssn_d1.lapse)

        # Term 1: K_ij contraction
        # Tensor block: T^{00} β^i β^j + 2 T^{0i} β^j + T^{ij}
        tensor_block = (
            TUU_00[:, None, None] * np.einsum('xi,xj->xij', beta_U, beta_U)
            + 2.0 * np.einsum('xi,xj->xij', TUU_0i, beta_U)
            + TUU_ij
        )

        term1_tau = np.einsum('xij,xij->x', K_LL, tensor_block)

        # Term 2: lapse derivative term ( line 352-357)
        # -(T^{00} β^i + T^{0i}) ∂_i alpha
        term2_tau = -(
            np.einsum('x,xi,xi->x', TUU_00, beta_U, dalpha_dx)
            + np.einsum('xi,xi->x', TUU_0i, dalpha_dx)
        )

        # Combine with volume element ( line 358)
        src_tau = alpha * e6phi * (term1_tau + term2_tau)

        # ====================================================================
        # MOMENTUM SOURCE TERM
        # ====================================================================

        # Shift derivatives ∂_i β^j (all spatial components)
        # Rescale derivatives: d(β^i)/dx^j = d(s_inv^i * shift_U^i)/dx^j
        dbeta_dx = (
            background.inverse_scaling_vector[:, :, None] * np.asarray(bssn_d1.shift_U)
            + bssn_vars.shift_U[:, :, None] * background.d1_inverse_scaling_vector
        )

        # Reference metric Christoffel symbols Γ̂^i_{jk}
        hat_chris = background.hat_christoffel

        # Covariant derivative of shift: \hat D_i β^j = ∂_i β^j + Γ̂^j_{ik} β^k
        #hatD_beta_U = dbeta_dx + np.einsum('xjik,xk->xij', hat_chris, beta_U)
        hatD_beta_U = np.transpose(dbeta_dx, (0, 2, 1)) + np.einsum('xjik,xk->xij', hat_chris, beta_U)

        # Covariant derivative of metric: ∇̂_i γ_{jk} ( line 407-415)
        dphi_dx = np.asarray(bssn_d1.phi)

        hat_D_bar_gamma = get_hat_D_bar_gamma_LL(r, bssn_vars.h_LL, bssn_d1.h_LL, background)

        # ∇̂_i γ_{jk} = e^{4φ} [4 γ̄_{jk} ∂_i φ + ∇̂_i γ̄_{jk}]
        # hat_D_bar_gamma has shape (N, j, k, i), need (N, i, j, k)
        hatD_gamma_LL = e4phi[:, None, None, None] * (
            4.0 * dphi_dx[:, :, None, None] * bar_gamma_LL[:, None, :, :]
            + np.transpose(hat_D_bar_gamma, (0, 3, 1, 2))
        )

        # Term 1: -T^{00} alpha ∂_i alpha ( line 418)
        # All three spatial components
        first_term = -TUU_00[:, None] * alpha[:, None] * dalpha_dx

        # Term 2: T^0_j ∇̂_i β^j ( line 420-423)
        # All three spatial components
        second_term = np.einsum('xj,xij->xi', TUD_0j, hatD_beta_U)

        # Term 3: (1/2) (T^{tt} β^j β^k + 2 T^{tj} β^k + T^{jk}) ∇̂_i γ_{jk}
        # This is the SAME tensor_block as in the energy equation ( line 425-433)
        # NOT just T^{jk}, but the full combination with shift terms
        tensor_block_momentum = (
            TUU_00[:, None, None] * np.einsum('xj,xk->xjk', beta_U, beta_U)
            + 2.0 * np.einsum('xj,xk->xjk', TUU_0i, beta_U)
            + TUU_ij
        )
        third_term = 0.5 * np.einsum('xjk,xijk->xi', tensor_block_momentum, hatD_gamma_LL)

        src_S_vector = alpha[:, None] * e6phi[:, None] * ( first_term + second_term + third_term)

        # Return full 3D momentum source vector and energy source
        return src_S_vector, src_tau  # (N, 3), (N,)


    def _compute_flux_derivative(self, F_D_face, F_S_face, F_tau_face):
        """
        Compute flux derivative contribution to Valencia equations using finite volumes.:
        Args:
            F_D_face: (N_faces,) density flux at interfaces
            F_S_face: (N_faces, 3) momentum flux at interfaces
            F_tau_face: (N_faces,) energy flux at interfaces

        Returns:
            der_D: (N,) flux divergence contribution to D
            der_S: (N, 3) flux divergence contribution to S_i
            der_tau: (N,) flux divergence contribution to tau
        """
        N = len(self.alpha)
        inv_dx = 1.0 / self.dx

        # Initialize with zeros (ghost cells remain zero)
        par_der_D = np.zeros(N)
        par_der_S = np.zeros((N, SPACEDIM))
        par_der_t = np.zeros(N)

        # Vectorized flux differences for interior cells
        i_s = NUM_GHOSTS
        i_e = N - NUM_GHOSTS

        par_der_D[i_s:i_e]    = - (F_D_face[i_s:i_e]    - F_D_face[i_s-1:i_e-1])    * inv_dx
        par_der_S[i_s:i_e, :] = - (F_S_face[i_s:i_e, :] - F_S_face[i_s-1:i_e-1, :]) * inv_dx
        par_der_t[i_s:i_e]    = - (F_tau_face[i_s:i_e]  - F_tau_face[i_s-1:i_e-1])  * inv_dx

        return par_der_D, par_der_S, par_der_t

    @staticmethod
    def _compute_fluxes(rho0, v_U, pressure, W, h, alpha, e6phi, gamma_LL, gamma_UU, beta_U):
        """
        Compute densitized partial flux vectors for the Valencia formulation.

        This is a static method that can be called from both Valencia class
        and external modules (e.g., riemann.py).

        Fluxes are densitized by e^{6φ}. vtilde^j = v^i - β^i/alpha.

        Flux definitions:
            F̃D^j    =  e^{6φ} ( rho0 * W * vtilde^j )
            F̃_S^j_i =  e^{6φ} ( alpha * T^j_i)
            F̃τ^j    =  e^{6φ} ( alpha^2 * T^{0j} - alpha * rho0 * W * vtilde^j )

        Returns:
            fD_U: (N, 3) density partial flux vector
            fTau_U: (N, 3) energy partial flux vector
            fS_D: (N, 3, 3) momentum partial flux tensor
        """
        # Compute 4-velocity: u^0 = W/alpha, u^i = W (v^i - β^i/alpha )
        u0U = W / alpha
        VUtilde_i = v_U - beta_U / alpha[:, None]
        uiU = W[:, None] * VUtilde_i
        # Compute contravariant stress-energy tensor T^{μν}
        # g^{μν} components from ADM variables
        alpha_sq = alpha[:, None] ** 2
        g4UU_0i = beta_U / alpha_sq
        g4UU_ij = gamma_UU - np.einsum('xi,xj->xij', beta_U, beta_U) / alpha_sq[:, None]

        # T^{0i} = ρ₀ h u^0 u^i + P g^{0i}
        rho_h = rho0 * h
        TUU_0i = (rho_h * u0U)[:, None] * uiU + pressure[:, None] * g4UU_0i

        # T^{ij} = ρ₀ h u^i u^j + P g^{ij}
        TUU_ij = rho_h[:, None, None] * np.einsum('xi,xj->xij', uiU, uiU) + pressure[:, None, None] * g4UU_ij

        # Compute mixed stress-energy tensor T^i_j = T^{ik} g_{kj}
        TUD_ij = np.einsum('xik,xkj->xij', TUU_ij, gamma_LL)

        # Conservative density: D = ρ₀ W
        D = rho0 * W

        # Density partial flux vector: F̃^j_D = e^6φ ρW (v^i - β^i/alpha )
        fD_U = (e6phi * alpha * D)[:, None] * VUtilde_i

        # Energy (tau) partial flux vector: F̃^j_τ = e^6φ ( alpha² T^{0j} - alpha * rho_0 W v^j )
        fTau_U = (alpha ** 2 * e6phi)[:, None] * TUU_0i -  (alpha * e6phi)[:, None] * D[:, None] * VUtilde_i

        # Momentum partial flux tensor: F̃^j_i = alpha e^6φ T^j_i
        fS_UD = (alpha * e6phi)[:, None, None] * TUD_ij

        return fD_U, fTau_U, fS_UD

    def _get_fluxes(self, rho0, v_U, pressure, W, h, bssn_vars):
        """
        Wrapper method to call _compute_fluxes with class geometry attributes.

        """
        phi_c = np.asarray(bssn_vars.phi, dtype=float)
        e6phi_c = np.exp(6.0 * phi_c)

        return self._compute_fluxes(rho0, v_U, pressure, W, h,
            self.alpha, e6phi_c, self.gamma_LL, self.gamma_UU, self.beta_U
        )

    def _compute_connection_terms(self, rho0, v_U, pressure, W, h, bssn_vars, background):
        """
        Compute connection term contributions from reference metric Christoffel symbols.

        Connection terms (form):
            D, τ:   -Γ̂^k_{kj} F̃^j
            S_i:    -Γ̂^k_{kj} F̃^j_i + Γ̂^l_{ji} F̃^j_l

        Where F̃^j are densitized partial fluxes ( e^{6φ} times physical flux).
        These terms arise from the covariant divergence in curved coordinates.

        Returns:
            conn_D: (N,) connection contribution to D equation
            conn_S: (N, 3) connection contribution to S_i equation
            conn_tau: (N,) connection contribution to tau equation
        """
        # Compute partial flux vectors (densitized)
        fD_U, fTau_U, fS_D = self._get_fluxes(rho0, v_U, pressure, W, h, bssn_vars)

        # Reference metric Christoffel symbols Γ̂^i_{jk}
        hat_chris = background.hat_christoffel  # (N, 3, 3, 3)

        # Trace: Γ̂^k_{kj} (sum over first index)
        Gamma_trace = np.einsum('xkkj->xj', hat_chris)

        # D equation: -Γ̂^k_{kj} F̃^j_D
        conn_D = -np.einsum('xj,xj->x', Gamma_trace, fD_U)

        # τ equation: -Γ̂^k_{kj} F̃^j_τ
        conn_tau = -np.einsum('xj,xj->x', Gamma_trace, fTau_U)

        # S_i equation: -Γ̂^k_{kj} F̃^j_i + Γ̂^l_{ji} F̃^j_l
        # First term: contract trace with momentum flux
        # Second term: full Christoffel contraction with momentum flux
        conn_S = (-np.einsum('xl,xli->xi', Gamma_trace, fS_D)
                  + np.einsum('xlji,xjl->xi', hat_chris, fS_D))

        return conn_D, conn_S, conn_tau

    def compute_rhs(self, D, S, tau, rho0, v_U, pressure, W, h,
                    r, bssn_vars, bssn_d1, background, spacetime_mode,
                    eos, grid, reconstructor, riemann_solver):
        """
        Compute RHS of Valencia equations - fully 3D implementation.

        Variables:
            D: (N,) - densitized rest-mass density
        S: (N,) or (N, 3) - densitized momentum (1D or 3D format)
        tau: (N,) - densitized energy
        v_U: (N,) or (N, 3) - spatial velocity (1D or 3D format)
        W: (N,) or None - Lorentz factor
        h: (N,) or None - specific enthalpy

        Returns:
        rhs_D: (N,) - time derivative of D
        rhs_S: (N, 3) - time derivative of momentum vector
        rhs_tau: (N,) - time derivative of tau
        """
        N = len(r)

        # Track original momentum dimensionality to preserve caller-facing API
        _S_input = np.asarray(S)
        _return_radial_only = (_S_input.ndim == 1) or (_S_input.ndim == 2 and _S_input.shape[1] == 1)

        # Coerce 1D inputs to 3D: (N,) -> (N, 3) with radial component only
        def to_3d(arr, N):
            if arr is None:
                return np.zeros((N, SPACEDIM))
            arr = np.asarray(arr, dtype=float)
            if arr.ndim == 1:
                out = np.zeros((N, SPACEDIM))
                out[:, 0] = arr
                return out
            return arr

        v_U = to_3d(v_U, N)
        S = to_3d(S, N)

        # Extract geometry (alpha, β^i, γ_ij, √γ, √ĝ, dr, etc.) early, as W may need γ_ij
        self._extract_geometry(r, bssn_vars, spacetime_mode, background, grid)

        # Copy conservatives
        D = D.copy()
        S = S.copy()
        tau = tau.copy()

        # Compute Fluxes at interfaces
        F_D_face, F_S_face, F_tau_face = self._compute_interface_fluxes(rho0, v_U, pressure, r, eos, reconstructor, riemann_solver, bssn_vars)

        # FLUX DERIVATIVE
        div_D, div_S, div_tau = self._compute_flux_derivative(F_D_face, F_S_face, F_tau_face)

        # CONNECTION TERMS FLUX derivative
        conn_D, conn_S, conn_tau = self._compute_connection_terms(rho0, v_U, pressure, W, h, bssn_vars, background)

        # SOURCE TERMS
        src_S, src_tau = self._compute_source_terms(rho0, v_U, pressure, W, h, bssn_vars, bssn_d1,background, spacetime_mode, r)

        # ====================================================================
        # ASSEMBLE RHS: divergence + connection + source
        # ====================================================================
        rhs_D = np.zeros(N)
        rhs_S = np.zeros((N, SPACEDIM))
        rhs_tau = np.zeros(N)

        # Interior cells only
        interior_mask = np.zeros(N, dtype=bool)
        interior_mask[NUM_GHOSTS:N-NUM_GHOSTS] = True

        rhs_D[interior_mask]    = (div_D[interior_mask]    + conn_D[interior_mask])

        rhs_S[interior_mask, :] = (div_S[interior_mask, :] + conn_S[interior_mask, :] + src_S[interior_mask, :])

        rhs_tau[interior_mask]  = (div_tau[interior_mask]  + conn_tau[interior_mask]  + src_tau[interior_mask])


        # Preserve backward-compatible shape: if input momentum was 1D, return RHS radial as (N,)
        if _return_radial_only:
            rhs_S_out = rhs_S[:, 0]
        else:
            rhs_S_out = rhs_S

        return rhs_D, rhs_S_out, rhs_tau  # (N,), (N,) or (N,3), (N,)

#------------------------------------------------------------------------------
# interface fluxes and reconstruction
#------------------------------------------------------------------------------
    def _compute_interface_fluxes(self, rho0, v_U, pressure, r,
                                eos, reconstructor, riemann_solver, bssn_vars):
        """
        Compute partial fluxes at cell interfaces: F̃_face = alpha_face √γ_face F_phys.

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
            Tuple of partial-flux arrays at interfaces (no √ĝ factor):
                F_D_face: (N_faces,) density flux
                F_S_face: (N_faces, 3) momentum flux (only [:, 0] non-zero)
                F_tau_face: (N_faces,) energy flux
        """
        N = len(r)

        # Extract first spatial component for 1D reconstruction
        v_r = v_U[:, 0]

        # Reconstruct primitives to left/right states at interfaces
        # Map Valencia boundary_mode to reconstructor boundary_type
        recon_boundary = "reflecting" if self.boundary_mode == "parity" else "outflow"
        (rhoL, vL, pL), (rhoR, vR, pR) = reconstructor.reconstruct_primitive_variables(
            rho0, v_r, pressure, x=r, boundary_type=recon_boundary
        )
        # ============================================================
        rhoL = np.maximum(rhoL, self.atmosphere.rho_floor)
        rhoR = np.maximum(rhoR, self.atmosphere.rho_floor)
        pL = np.maximum(pL, self.atmosphere.p_floor)
        pR = np.maximum(pR, self.atmosphere.p_floor)
        # ============================================================

        # Extract interior faces
        rhoL, vL, pL = rhoL[1:-1], vL[1:-1], pL[1:-1]
        rhoR, vR, pR = rhoR[1:-1], vR[1:-1], pR[1:-1]

        # Interpolate geometry to faces from class attributes
        alpha_f = 0.5 * (self.alpha[:-1] + self.alpha[1:])
        # Exact e^{6phi} at faces from BSSN phi
        phi_arr = np.asarray(bssn_vars.phi, dtype=float)
        phi_face = 0.5 * (phi_arr[:-1] + phi_arr[1:])
        e6phi_f = np.exp(6.0 * phi_face)

        # Interpolate shift and metric to faces
        beta_U_f = 0.5 * (self.beta_U[:-1] + self.beta_U[1:])  # (N-1, 3)
        gamma_LL_f = 0.5 * (self.gamma_LL[:-1] + self.gamma_LL[1:])  # (N-1, 3, 3)

        # For 1D Riemann solver, extract first spatial components
        beta_r_f = beta_U_f[:, 0]  # β^r
        gamma_r_r_f = gamma_LL_f[:, 0, 0]  # γ_rr

        # Create GeometryState for cell faces (used by prim_to_cons and riemann solver)
        geom_f = GeometryState(
            alpha=alpha_f,
            beta_r=beta_r_f,
            gamma_rr=gamma_r_r_f,
            e6phi=e6phi_f
        )

        # Apply physical limiters if available
        if hasattr(reconstructor, "apply_physical_limiters"):
            (rhoL, vL, pL), (rhoR, vR, pR) = reconstructor.apply_physical_limiters(
                (rhoL, vL, pL), (rhoR, vR, pR),
                atmosphere=self.atmosphere,
                gamma_rr=gamma_r_r_f  # More general name
            )

        # Convert primitives to conservatives at interfaces
        # prim_to_cons computes: D, S_r, τ where S_r is momentum
        UL_D, UL_S_r, UL_tau = prim_to_cons(rhoL, vL, pL, geom_f, eos)
        UR_D, UR_S_r, UR_tau = prim_to_cons(rhoR, vR, pR, geom_f, eos)

        # Package for Riemann solver
        UL_batch = np.stack([UL_D, UL_S_r, UL_tau], axis=1)
        UR_batch = np.stack([UR_D, UR_S_r, UR_tau], axis=1)
        primL_batch = np.stack([rhoL, vL, pL], axis=1)
        primR_batch = np.stack([rhoR, vR, pR], axis=1)

        # Solve Riemann problem to get physical fluxes
        # Returns: [F_D, F_S_r, F_tau] in r spatial direction
        F_phys_batch = riemann_solver.solve_batch(UL_batch, UR_batch, primL_batch, primR_batch, geom_f, eos)


        F_batch = F_phys_batch

        # Construct 3D momentum flux array
        # In 1D evolution: only r direction has flux, others are zero

        N_faces = len(F_batch)
        F_S_face = np.zeros((N_faces, SPACEDIM))
        F_S_face[:, 0] = F_batch[:, 1]  # r direction flux
        # F_S_face[:, 1] = 0  # Secondary direction (no flux in 1D)
        # F_S_face[:, 2] = 0  # Tertiary direction (no flux in 1D)

        return F_batch[:, 0], F_S_face, F_batch[:, 2]

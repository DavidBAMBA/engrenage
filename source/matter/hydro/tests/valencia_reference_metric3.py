# valencia_reference_metric.py
"""
Valencia formulation with reference metric - full 3D BSSN-style implementation.

Follows the exact same tensor algebra pattern as bssnrhs.py:
- Full 3D einsum contractions for all tensor operations
- Spherical symmetry imposed by velocity (v^θ = v^φ = 0) and background metric
- Source terms following  + GRHD_equations.py structure
- Connection terms with CORRECT signs for covariant divergence

Conservative evolution equations in curved coordinates (two equivalent views):
    Densitized form:  ∂_t(U) + (1/√ĝ) ∂_j[√ĝ F^j] = S
      form used here:  ∂_t(U) + ∂_j(F̃^j) = S + connection
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
from source.core.spacing import NUM_GHOSTS
from source.matter.hydro.cons2prim import prim_to_cons


class StressEnergyTensor4D:
    """
    Efficient container for 4D stress-energy tensor components.
    Replaces dict for better performance (eliminates dict overhead and hash lookups).

    Attributes:
        T00: Temporal-temporal component (N,)
        T0i: Temporal-spatial components (N, 3)
        Tij: Spatial-spatial components (N, 3, 3)
    """
    __slots__ = ('T00', 'T0i', 'Tij')

    def __init__(self, N):
        """Preallocate arrays for N grid points."""
        self.T00 = np.zeros(N)
        self.T0i = np.zeros((N, SPACEDIM))
        self.Tij = np.zeros((N, SPACEDIM, SPACEDIM))


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

        # Geometric quantities
        # These will be computed by _extract_geometry()
        self.alpha = None              # Lapse function
        self.beta_U = None             # Shift vector (N, 3)
        self.gamma_LL = None           # Physical metric (N, 3, 3)
        self.gamma_UU = None           # Inverse metric (N, 3, 3)
        self.sqrt_gamma = None         # √γ determinant factor
        self.e6phi = None              # e^(6φ) conformal factor
        self.sqrt_g_hat_cell = None    # √ĝ at cell centers
        self.sqrt_g_hat_face = None    # √ĝ at cell faces
        self.dr = None                 # Mesh spacing

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
        N = len(r)

        # Mesh spacing (computed once per timestep)
        if grid is not None and hasattr(grid, 'derivs') and hasattr(grid.derivs, 'dx'):
            self.dr = float(grid.derivs.dx)
        elif grid is not None and hasattr(grid, 'dr'):
            self.dr = float(grid.dr)
        elif grid is not None and hasattr(grid, 'spacing') and np.isscalar(grid.spacing):
            self.dr = float(grid.spacing)
        elif grid is not None and hasattr(grid, 'spacing') and hasattr(grid.spacing, '__len__'):
            # For 3D grids with multiple spacings, return radial spacing
            self.dr = float(grid.spacing[0])
        else:
            # Fallback: compute from radial coordinate array (used when grid is None)
            self.dr = float(r[1] - r[0]) if len(r) > 1 else 1.0

        if spacetime_mode == "fixed_minkowski":
            self.alpha = np.ones(N)

            # All three shift components are zero
            self.beta_U = np.zeros((N, SPACEDIM))

            # Identity metric in 3D
            self.gamma_LL = np.zeros((N, SPACEDIM, SPACEDIM))
            for i in range(SPACEDIM):
                self.gamma_LL[:, i, i] = 1.0

            # Inverse metric (also identity)
            self.gamma_UU = np.copy(self.gamma_LL)

            # Determinant and square root
            self.sqrt_gamma = np.ones(N)
            self.sqrt_g_hat_cell = background.det_hat_gamma ** 0.5
            self.sqrt_g_hat_face = 0.5 * (self.sqrt_g_hat_cell[:-1] + self.sqrt_g_hat_cell[1:])

        else:
            # Lapse α
            self.alpha = np.asarray(bssn_vars.lapse, dtype=float)

            # Shift β^i (all three components)
            self.beta_U = np.zeros((N, SPACEDIM))

            if hasattr(bssn_vars, 'shift_U') and bssn_vars.shift_U is not None:
                shift_array = np.asarray(bssn_vars.shift_U)
                if shift_array.ndim >= 2:
                    for i in range(min(SPACEDIM, shift_array.shape[1])):
                        self.beta_U[:, i] = shift_array[:, i].astype(float)

            self.beta_U = self.beta_U * background.inverse_scaling_vector


            # Conformal factor e^{φ}
            phi_arr = np.asarray(bssn_vars.phi, dtype=float)
            e6phi = np.exp(6.0 * phi_arr)
            e4phi = np.exp(4.0 * phi_arr)

            # Physical metric γ_ij = e^{4φ} γ̄_ij (full 3x3 tensor)
            bar_gamma_LL = get_bar_gamma_LL(r, bssn_vars.h_LL, background)
            self.gamma_LL = e4phi[:, None, None] * bar_gamma_LL

            # Inverse physical metric γ^{ij}
            self.gamma_UU = np.linalg.inv(self.gamma_LL)

            # √γ = e^{6φ} γ̄ = γ̂
            self.sqrt_gamma = e6phi
            self.e6phi = e6phi 

            # √ĝ for reference metric
            self.sqrt_g_hat_cell = np.sqrt(np.abs(background.det_hat_gamma) + 1e-30)
            self.sqrt_g_hat_face = 0.5 * (self.sqrt_g_hat_cell[:-1] + self.sqrt_g_hat_cell[1:])

    def _compute_4velocity(self, rho0, v_U, W):
        """
        Compute 4-velocity components u^μ 

        u^0 = W/α (timelike component, Valencia formulation)
        u^i = W v^i (spatial components)

        Args:
            v_U: (N, 3) array with [v^x, v^y, v^z] spatial velocity components

        Returns: u^μ as (N, 4) array [u^t, u^x, u^y, u^z]
        """
        N = len(rho0)

        # 4-velocity components 
        u4U = np.zeros((N, 4))
        u4U[:, 0] = W / self.alpha  # u^t = u^0

        # All three spatial components
        for i in range(SPACEDIM):
            u4U[:, i + 1] = W * v_U[:, i]  # u^i = W v^i

        return u4U

    def _compute_T4UU(self, rho0, v_U, pressure, W, h):
        """
        Compute contravariant stress-energy tensor T^{μν} 

        T^{μν} = ρ₀ h u^μ u^ν + P g^{μν}

        Args:
            v_U: (N, 3) array with [v^x, v^y, v^z] spatial velocity components

        Returns: T^{μν} with components:
            - T^{00} = T^{tt} (energy density in Eulerian frame)
            - T^{0i} = T^{ti} (energy flux / momentum density)
            - T^{ij} (momentum flux / stress tensor)
        """
        N = len(rho0)

        # Get 4-velocity
        u4U = self._compute_4velocity(rho0, v_U, W)

        # Compute g^{μν} components 
        # g^{tt} = -1/α²
        g4UU_00 = -1.0 / (self.alpha ** 2)

        # g^{ti} = β^i/α²
        g4UU_0i = self.beta_U / (self.alpha[:, None] ** 2)

        # g^{ij} = γ^{ij} - β^i β^j / α²
        g4UU_spatial = self.gamma_UU - np.einsum('xi,xj->xij', self.beta_U, self.beta_U) / (self.alpha[:, None, None] ** 2)

        # Compute T^{μν} = ρ₀ h u^μ u^ν + P g^{μν} ( + line 159-161)
        T4UU = StressEnergyTensor4D(N)

        # T^{00} = ρ₀ h u^0 u^0 + P g^{00}
        T4UU.T00[:] = rho0 * h * u4U[:, 0] * u4U[:, 0] + pressure * g4UU_00

        # T^{0i} = ρ₀ h u^0 u^i + P g^{0i} (all three spatial components)
        for i in range(SPACEDIM):
            T4UU.T0i[:, i] = (
                rho0 * h * u4U[:, 0] * u4U[:, i + 1]
                + pressure * g4UU_0i[:, i]
            )

        # T^{ij} = ρ₀ h u^i u^j + P g^{ij} (all nine spatial components)
        for i in range(SPACEDIM):
            for j in range(SPACEDIM):
                T4UU.Tij[:, i, j] = (
                    rho0 * h * u4U[:, i + 1] * u4U[:, j + 1]
                    + pressure * g4UU_spatial[:, i, j]
                )

        return T4UU

    def _compute_T4UD(self, T4UU):
        """
        Compute mixed stress-energy tensor T^μ_ν = T^{μδ} g_{δν} following  + (line 176-198).

        Needed for momentum flux computation.

        Args:
            T4UU: Contravariant stress-energy tensor from _compute_T4UU

        Returns: T^μ_ν with all components
        """
        N = len(self.alpha)

        # Geometric quantities from class attributes
        alpha = self.alpha
        beta_U = self.beta_U  # (N, SPACEDIM)
        gamma_LL = self.gamma_LL  # (N, SPACEDIM, SPACEDIM)
        
        # Build g_{μν} components ( + uses ADM_to_g4DD)
        # g_{tt} = -α² + β_k β^k (using γ_ij to lower β)
        beta_lower = np.einsum('xij,xj->xi', gamma_LL, beta_U)
        beta_squared = np.einsum('xi,xi->x', beta_U, beta_lower)
        g4DD_00 = -alpha ** 2 + beta_squared
        
        # g_{ti} = β_i (lowered with γ)
        g4DD_0i = beta_lower
        
        # g_{ij} = γ_{ij}
        g4DD_spatial = gamma_LL

        # Compute T^μ_ν = T^{μδ} g_{δν} ( + line 194-198)
        T4UD = StressEnergyTensor4D(N)

        # T^0_0 = T^{00} g_{00} + T^{0i} g_{i0}
        T4UD.T00[:] = (
            T4UU.T00 * g4DD_00
            + np.einsum('xi,xi->x', T4UU.T0i, g4DD_0i)
        )

        # T^0_j = T^{00} g_{0j} + T^{0i} g_{ij} (all three components)
        T4UD.T0i[:] = (
            T4UU.T00[:, None] * g4DD_0i
            + np.einsum('xi,xij->xj', T4UU.T0i, g4DD_spatial)
        )

        # T^i_j = T^{i0} g_{0j} + T^{ik} g_{kj} (full computation,  + line 194-198)
        for i in range(SPACEDIM):
            T4UD.Tij[:, i, :] = (
                T4UU.T0i[:, i, None] * g4DD_0i
                + np.einsum('xk,xkj->xj', T4UU.Tij[:, i, :], g4DD_spatial)
            )

        return T4UD

    def _compute_source_terms(self, rho0, v_U, pressure, W, h,
                            bssn_vars, bssn_d1, background, spacetime_mode, r):
        """
        Physical source terms of GRHD equations in Fully 3D implementation.

        Energy source (tau_source_term,  + line 334-358):
            S_τ = α e^{6φ} [K_ij (T^{00} β^i β^j + 2 T^{0i} β^j + T^{ij})
                                    - (T^{00} β^i + T^{0i}) ∂_i α]

        Momentum source S_tilde_source_termD:
            S_{S_i} = α e^{6φ}[-T^{00} α ∂_i α
                                        + T^0_j (∂_i β^j + Γ̂^j_{ik} β^k)
                                        + (1/2) (T^{00} β^j β^k + 2 T^{tj} β^k + T^{jk}) ∇̂_i γ_{jk}]

        Args:
            v_U: (N, 3) array with [v^x, v^y, v^z] spatial velocity components

        Returns:
            src_S_vector: (N, 3) momentum source for all three spatial directions
            src_tau: (N,) energy source
        """
        N = len(r)

        # Handle Minkowski case separately: no explicit hoop-stress sources here.
        # In the   formulation, geometric effects (e.g., 2p/r) arise from connection terms,
        # not explicit sources. So in fixed_minkowski we set sources to zero and rely on
        # the connection block in compute_rhs.
        if spacetime_mode == "fixed_minkowski":
            src_S_vector = np.zeros((N, SPACEDIM))
            src_tau = np.zeros(N)
            return src_S_vector, src_tau

        # Geometric quantities from class attributes
        alpha = self.alpha
        beta_U = self.beta_U  # (N, SPACEDIM) - all three components
        
        # Compute e^{6φ} and √(det γ̄) separately for clarity
        # Use only e^{6φ} as volume element (GRoovy Eq. 20)
        # The √(det γ̂) factor is implicitly handled via Γ̂^i_{jk}
        phi = np.asarray(bssn_vars.phi, dtype=float)
        e6phi = np.exp(6.0 * phi)

        
        # Build full 3D metric γ_ij = e^{4φ} γ̄_ij
        e4phi = np.exp(4.0 * phi)
        bar_gamma_LL = get_bar_gamma_LL(r, bssn_vars.h_LL, background)
        gamma_LL = e4phi[:, None, None] * bar_gamma_LL
        
        # Compute stress-energy tensors T^{μν} and T^μ_ν with all three velocity components
        T4UU = self._compute_T4UU(rho0, v_U, pressure, W, h)
        T4UD = self._compute_T4UD(T4UU)
        
        # ====================================================================
        # ENERGY SOURCE TERM 
        # ====================================================================
        
        # Extrinsic curvature K_ij = e^{4φ} Ā_ij + (K/3) γ_ij
        K = np.asarray(bssn_vars.K, dtype=float)
        bar_A_LL = get_bar_A_LL(r, bssn_vars, background)
        K_LL = e4phi[:, None, None] * bar_A_LL + (K / 3.0)[:, None, None] * gamma_LL
        
        # Lapse derivatives ∂_i α (all three spatial directions)
        dalpha_dx = np.zeros((N, SPACEDIM))
        if hasattr(bssn_d1, 'lapse') and bssn_d1.lapse is not None:
            dalpha_dx = np.asarray(bssn_d1.lapse)
        
        # Term 1: K_ij contraction 
        # Tensor block: T^{00} β^i β^j + 2 T^{0i} β^j + T^{ij}
        tensor_block_energy = (
            T4UU.T00[:, None, None] * np.einsum('xi,xj->xij', beta_U, beta_U)
            + 2.0 * np.einsum('xi,xj->xij', T4UU.T0i, beta_U)
            + T4UU.Tij
        )

        term1_tau = np.einsum('xij,xij->x', K_LL, tensor_block_energy)

        # Term 2: lapse derivative term
        # -(T^{00} β^i + T^{0i}) ∂_i α
        term2_tau = -(
            np.einsum('x,xi,xi->x', T4UU.T00, beta_U, dalpha_dx)
            + np.einsum('xi,xi->x', T4UU.T0i, dalpha_dx)
        )
        
        # Final energy source term
        src_tau = alpha * e6phi * (term1_tau + term2_tau)
        
        # ====================================================================
        # MOMENTUM SOURCE TERM 
        # ====================================================================

        # Shift derivatives ∂_i β^j (all spatial components)
        dbeta_dx = np.zeros((N, SPACEDIM, SPACEDIM))
        if hasattr(bssn_d1, 'shift_U') and bssn_d1.shift_U is not None:
            shift_d1 = np.asarray(bssn_d1.shift_U)
            if shift_d1.ndim >= 3:
                # Las derivadas del shift también necesitan reescalamiento
                # d(β^i)/dx^j = d(s^i * shift^i)/dx^j = s^i * d(shift^i)/dx^j + shift^i * d(s^i)/dx^j
                for i in range(SPACEDIM):
                    for j in range(SPACEDIM):
                        dbeta_dx[:, i, j] = (
                            background.inverse_scaling_vector[:, i] * shift_d1[:, i, j]
                            + bssn_vars.shift_U[:, i] * background.d1_inverse_scaling_vector[:, i, j]
                        )

        # Reference metric Christoffel symbols Γ̂^i_{jk}
        hat_chris = background.hat_christoffel

        # Covariant derivative of shift: ∇̂_i β^j = ∂_i β^j + Γ̂^j_{ik} β^k
        # ( + line 422-423)
        hatD_beta_U = dbeta_dx + np.einsum('xjik,xk->xij', hat_chris, beta_U)

        # Covariant derivative of metric: ∇̂_i γ_{jk} ( + line 407-415)
        dphi_dx = np.zeros((N, SPACEDIM))
        if hasattr(bssn_d1, 'phi') and bssn_d1.phi is not None:
            dphi_dx = np.asarray(bssn_d1.phi)

        hat_D_bar_gamma = get_hat_D_bar_gamma_LL(r, bssn_vars.h_LL, bssn_d1.h_LL, background)

        # ∇̂_i γ_{jk} = e^{4φ} [4 γ̄_{jk} ∂_i φ + ∇̂_i γ̄_{jk}]
        hatD_gamma_LL = np.zeros((N, SPACEDIM, SPACEDIM, SPACEDIM))
        for i in range(SPACEDIM):
            for j in range(SPACEDIM):
                for k in range(SPACEDIM):
                    # Phi term + covariant derivative of \bar{\gamma}_{jk}
                    hatD_gamma_LL[:, i, j, k] = (
                        4.0 * bar_gamma_LL[:, j, k] * dphi_dx[:, i]
                        + hat_D_bar_gamma[:, j, k, i]
                    )
                    # Multiply by e^{4φ} to get physical metric derivative
                    hatD_gamma_LL[:, i, j, k] *= e4phi

        # Term 1: -T^{00} α ∂_i α ( + line 418)
        # All three spatial components
        first_term = -T4UU.T00[:, None] * alpha[:, None] * dalpha_dx

        # Term 2: T^0_j ∇̂_i β^j 
        # All three spatial components
        second_term = np.einsum('xj,xij->xi', T4UD.T0i, hatD_beta_U)

        # Term 3: (1/2) (T^{tt} β^j β^k + 2 T^{0j} β^k + T^{jk}) ∇̂_i γ_{jk}
        # This is the SAME tensor_block as in the energy equation ( + line 425-433)
        tensor_block_momentum = (
            T4UU.T00[:, None, None] * np.einsum('xj,xk->xjk', beta_U, beta_U)
            + 2.0 * np.einsum('xj,xk->xjk', T4UU.T0i, beta_U)
            + T4UU.Tij
        )
        third_term = 0.5 * np.einsum('xjk,xijk->xi', tensor_block_momentum, hatD_gamma_LL)

        # Combine with volume element ( + line 436-438)
        # This gives all three spatial components of the momentum source
        src_S_vector = alpha[:, None] * e6phi[:, None] * (
            first_term + second_term + third_term
        )

        # Return full 3D momentum source vector and energy source
        return src_S_vector, src_tau  # (N, 3), (N,)

    def compute_rhs(self, D, S_tildeD, tau, rho0, v_U, pressure, W, h,
                    r, bssn_vars, bssn_d1, background, spacetime_mode,
                    eos, grid, reconstructor, riemann_solver):
        """
        Compute RHS of Valencia equations - fully 3D implementation.

        Follows BSSN pattern: always requires 3D vector inputs for velocity and momentum.
        Spherical symmetry is enforced by background geometry (Christoffel symbols)
        and boundary conditions, NOT by restricting array dimensions.

        Variables:
            D: (N,) - densitized rest-mass density
            S_tildeD: (N, 3) - densitized momentum (full 3D vector)
            tau: (N,) - densitized energy
            v_U: (N, 3) - spatial velocity (full 3D vector)
            W: (N,) or None - Lorentz factor (computed if None)
            h: (N,) or None - specific enthalpy (computed if None)

        Returns:
            rhs_D: (N,) - time derivative of D
            rhs_S_tildeD: (N, 3) - time derivative of momentum vector (always 3D)
            rhs_tau: (N,) - time derivative of tau
        """
        N = len(r)

        # ====================================================================
        # INPUT VALIDATION: Enforce 3D shapes (following BSSN pattern)
        # ====================================================================
        # Unlike legacy API, we require fully 3D inputs (N, 3) for v_U and S_tildeD.
        # Spherical symmetry is enforced by the background geometry (Christoffel symbols)
        # and boundary conditions, NOT by restricting array dimensions to 1D.
        v_U = np.asarray(v_U)
        S_tildeD = np.asarray(S_tildeD)

        if v_U.shape != (N, SPACEDIM):
            raise ValueError(f"v_U must have shape ({N}, {SPACEDIM}), got {v_U.shape}. "
                           f"Pass full 3D velocity vector (zeros for angular components in spherical symmetry).")
        if S_tildeD.shape != (N, SPACEDIM):
            raise ValueError(f"S_tildeD must have shape ({N}, {SPACEDIM}), got {S_tildeD.shape}. "
                           f"Pass full 3D momentum vector (zeros for angular components in spherical symmetry).")

        # ====================================================================
        # SPHERICAL SYMMETRY: Enforce v^r = 0 at origin (first interior cell)
        # ====================================================================
        # By spherical symmetry, radial velocity MUST be exactly zero at r=0.
        # Even tiny numerical drift (v^r ~ 1e-6) gets amplified by large
        # Christoffel symbols (Γ ~ 200/M) near origin, creating spurious
        # connection terms that act as mass sources.
        #
        # IMPORTANT: Only apply in true spherical spacetimes, NOT in Minkowski mode
        # where there is no physical origin (Sod test, Cartesian problems, etc.)
        if self.boundary_mode == "parity" and spacetime_mode != "fixed_minkowski":
            v_U[NUM_GHOSTS, 0] = 0.0  # Force v^r = 0 at r ≈ 0

        # Extract geometry (α, β^i, γ_ij, √γ, √ĝ, dr, etc.) early, as W may need γ_ij
        self._extract_geometry(r, bssn_vars, spacetime_mode, background, grid)

        # Compute W and h from primitives if not provided
        if W is None:
            # v² = γ_ij v^i v^j
            v_squared = np.einsum('xij,xi,xj->x', self.gamma_LL, v_U, v_U)
            # Lorentz factor: W = 1/√(1 - v²)
            W = 1.0 / np.sqrt(np.maximum(1.0 - v_squared, 1e-16))

        if h is None:
            eps = eos.eps_from_rho_p(rho0, pressure)
            h = 1.0 + eps + pressure / np.maximum(rho0, 1e-30)

        # Copy conservatives (boundary conditions handled by Grid.fill_boundaries)
        D = D.copy()
        S_tildeD = S_tildeD.copy()
        tau = tau.copy()

        # Compute  -style partial fluxes at interfaces using α e^{6φ} at faces
        F_D_face, F_S_tildeD_face, F_tau_face = self._compute_interface_fluxes(
            rho0, v_U, pressure, r, eos, reconstructor, riemann_solver, bssn_vars, spacetime_mode
        )

        # Reference metric Christoffel symbols Γ̂^i_{jk}
        hat_chris = background.hat_christoffel  # (N, 3, 3, 3)

        # ====================================================================
        # FLUX DIVERGENCE (  form): -∂_r(F̃^r)
        #   with F̃^r = α e^{6φ}* F_phys^r (no √ĝ factor here)
        # ====================================================================

        # Use the new vectorized divergence computation
        rhs_D, rhs_S_tildeD, rhs_tau = compute_divergence_vectorized(
            F_D_face, F_S_tildeD_face, F_tau_face, self.dr
        )

        # ====================================================================
        # CONNECTION TERMS (  form):
        #   D, τ:      -Γ̂^k_{kr} F̃^r
        #   S_i:       -Γ̂^k_{kr} F̃^r_i + Γ̂^l_{ri} F̃^r_l
        # ====================================================================

        alpha = self.alpha
        phi = np.asarray(bssn_vars.phi, dtype=float)
        e6phi = np.exp(6.0 * phi)


        # Compute 4-velocity and stress-energy
        u4U = self._compute_4velocity(rho0, v_U, W)
        T4UU = self._compute_T4UU(rho0, v_U, pressure, W, h)

        # Conservative density
        rho_star = alpha * e6phi * rho0 * u4U[:, 0]

        # Physical fluxes at cell centers
        fD_U = np.zeros((N, SPACEDIM))
        for j in range(SPACEDIM):
            fD_U[:, j] = rho_star * v_U[:, j]

        # Energy (tau) partial flux vector, densitized:
        #   F̃^j_tau = α e^{6φ} ( α T^{0j} ) - ρ_* v^j
        fTau_U = np.zeros((N, SPACEDIM))
        for j in range(SPACEDIM):
            fTau_U[:, j] = alpha ** 2 * e6phi * T4UU.T0i[:, j] - rho_star * v_U[:, j]

        # Momentum partial flux tensor F̃^j_i = α e^{6φ} T^j_i
        T4UD = self._compute_T4UD(T4UU)
        F_S_no_ghat = np.zeros((N, SPACEDIM, SPACEDIM))
        for j in range(SPACEDIM):
            for i in range(SPACEDIM):
                F_S_no_ghat[:, j, i] = alpha * e6phi * T4UD.Tij[:, j, i]

        Gamma_trace = np.einsum('xkkj->xj', hat_chris)


        # Connection contributions
        conn_D   = -np.einsum('xj,xj->x', Gamma_trace, fD_U)
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
        # SOURCE TERMS (geometric couplings -  + style)
        # ====================================================================
        src_S_tildeD, src_tau = self._compute_source_terms(
            rho0, v_U, pressure, W, h, bssn_vars, bssn_d1,
            background, spacetime_mode, r
        )  # Returns (N,3), (N,)

        rhs_S_tildeD += src_S_tildeD  # Vector addition - all three components
        rhs_tau += src_tau

        return rhs_D, rhs_S_tildeD, rhs_tau  # Always (N,), (N,3), (N,)

    def _compute_interface_fluxes(self, rho0, v_U, pressure, r,
                                eos, reconstructor, riemann_solver, bssn_vars, spacetime_mode):
        """
        Compute  -style partial fluxes at cell interfaces: F̃_face = α_face √γ_face F_phys.

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
                F_S_tildeD_face: (N_faces, 3) momentum flux (only [:, 0] non-zero)
                F_tau_face: (N_faces,) energy flux
        """
        N = len(r)

        # Extract first spatial component for 1D reconstruction
        v_primary = v_U[:, 0]

        # Reconstruct primitives to left/right states at interfaces
        # Map Valencia boundary_mode to reconstructor boundary_type
        # IMPORTANT: Only use reflecting BC in true spherical spacetimes, not in Minkowski
        use_reflecting = (self.boundary_mode == "parity" and spacetime_mode != "fixed_minkowski")
        recon_boundary = "reflecting" if use_reflecting else "outflow"
        (rhoL, vL, pL), (rhoR, vR, pR) = reconstructor.reconstruct_primitive_variables(
            rho0, v_primary, pressure, x=r, boundary_type=recon_boundary
        )

        # Extract interior faces (exclude ghost zones)
        rhoL, vL, pL = rhoL[1:-1], vL[1:-1], pL[1:-1]
        rhoR, vR, pR = rhoR[1:-1], vR[1:-1], pR[1:-1]

        # Interpolate geometry to faces from class attributes
        alpha_f = 0.5 * (self.alpha[:-1] + self.alpha[1:])
        sqrt_gamma_f = 0.5 * (self.sqrt_gamma[:-1] + self.sqrt_gamma[1:])
        sqrt_g_hat_f = self.sqrt_g_hat_face
        # Exact e^{6phi} at faces from BSSN phi ( -compatible densitization)
        phi_arr = np.asarray(bssn_vars.phi, dtype=float)
        phi_face = 0.5 * (phi_arr[:-1] + phi_arr[1:])
        e6phi_f = np.exp(6.0 * phi_face)

        # Interpolate shift and metric to faces
        beta_U_f = 0.5 * (self.beta_U[:-1] + self.beta_U[1:])  # (N-1, 3)
        gamma_LL_f = 0.5 * (self.gamma_LL[:-1] + self.gamma_LL[1:])  # (N-1, 3, 3)
        
        # For 1D Riemann solver, extract first spatial components
        beta_primary_f = beta_U_f[:, 0]  # β^r or β^ρ or β^x
        gamma_primary_primary_f = gamma_LL_f[:, 0, 0]  # γ_rr or γ_ρρ or γ_xx
        
        # Enforce reflecting condition at the r≈0 interface
        # After extraction, index 0 corresponds to the face between
        # the last ghost cell and first interior cell
        if recon_boundary == "reflecting" and (len(vL) > 0):
            # Enforce v=0 at the innermost face
            vL[0] = 0.0
            vR[0] = 0.0
            # For density and pressure, use symmetric values
            # This enforces even parity for these quantities
            if len(rho0) > NUM_GHOSTS:
                # Use first interior value for consistency
                rho_ref = rho0[NUM_GHOSTS]
                p_ref = pressure[NUM_GHOSTS]
                rhoL[0] = rho_ref
                rhoR[0] = rho_ref
                pL[0] = p_ref
                pR[0] = p_ref
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
        
        # Densitization strictly α e^{6φ} at faces
        dens_factor = alpha_f * e6phi_f

        F_batch = dens_factor[:, None] * F_phys_batch

        # FIX: Enforce zero momentum flux at the r≈0 interface
        # After extraction (lines 797-798), index 0 in F_batch corresponds to
        # the face between the last ghost cell and first interior cell
        #
        # IMPORTANT: Only apply in true spherical spacetimes, NOT in Minkowski mode
        # where there is no physical origin (Sod test, Cartesian problems, etc.)
        if self.boundary_mode == "parity" and spacetime_mode != "fixed_minkowski":
            # The first face must have zero radial momentum flux
            # to enforce the reflecting boundary condition at r=0
            if len(F_batch) > 0:
                F_batch[0, 1] = 0.0  # F_Sr = 0 at the r≈0 interface

        # Additional freeze: if one side of a face is pure atmosphere,
        # zero all hydro fluxes across that face (TOV surface handling).
        # This prevents spurious mass/momentum leakage into vacuum in static stars.
        try:
            rho_atm_face = self.atmosphere.rho_floor * getattr(self.atmosphere, 'rho_threshold', 1.0)
            # Faces arrays align with reconstructed states (after trimming ghosts)
            # We have access to rhoL, rhoR defined above (already trimmed to faces)
            mask_freeze = (rhoL < rho_atm_face) | (rhoR < rho_atm_face)
            if np.any(mask_freeze) and spacetime_mode != "fixed_minkowski":
                F_batch[mask_freeze, :] = 0.0
        except Exception:
            pass

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

        return F_batch[:, 0], F_S_tildeD_face, F_batch[:, 2]


# ============================================================================
# MODULAR FUNCTIONS FOR FINITE VOLUME SCHEME (solver.py style)
# ============================================================================

from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class ValenciaGeometry:
    """Container for geometric quantities at cell centers and faces."""
    # Cell-centered quantities
    alpha: np.ndarray          # Lapse at cells
    beta_U: np.ndarray         # Shift at cells (N, 3)
    gamma_LL: np.ndarray       # Metric at cells (N, 3, 3)
    gamma_UU: np.ndarray       # Inverse metric at cells (N, 3, 3)
    sqrt_gamma: np.ndarray     # √γ at cells
    e6phi: np.ndarray          # e^{6φ} at cells
    sqrt_g_hat_cell: np.ndarray  # √ĝ at cells
    dr: float                  # Radial spacing

    # Face quantities (interpolated) - optional
    alpha_f: Optional[np.ndarray] = None
    beta_U_f: Optional[np.ndarray] = None
    gamma_LL_f: Optional[np.ndarray] = None
    e6phi_f: Optional[np.ndarray] = None
    sqrt_g_hat_face: Optional[np.ndarray] = None


def compute_divergence_vectorized(F_D_face: np.ndarray,
                                 F_S_face: np.ndarray,
                                 F_tau_face: np.ndarray,
                                 dr: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute flux divergence using vectorized operations.

    Replaces the for-loop implementation with NumPy array slicing for better performance.
    Formula: -∂_r(F̃^r) = -(F̃^r_{i+1/2} - F̃^r_{i-1/2}) / Δr

    Args:
        F_D_face: Density flux at cell faces (N_faces,)
        F_S_face: Momentum flux at cell faces (N_faces, 3)
        F_tau_face: Energy flux at cell faces (N_faces,)
        dr: Radial grid spacing

    Returns:
        rhs_D: Density RHS contribution from flux divergence (N,)
        rhs_S: Momentum RHS contribution from flux divergence (N, 3)
        rhs_tau: Energy RHS contribution from flux divergence (N,)
    """
    # Determine number of cells from number of faces
    N = F_D_face.shape[0] + 1

    # Initialize RHS arrays
    rhs_D = np.zeros(N)
    rhs_S = np.zeros((N, SPACEDIM))
    rhs_tau = np.zeros(N)

    # Compute divergence using vectorized operations for interior cells
    # Note: F_face[i] is the flux at the right interface of cell i
    inv_dr = 1.0 / (dr + 1e-30)

    # Interior cells: NUM_GHOSTS <= i < N - NUM_GHOSTS
    i_start = NUM_GHOSTS
    i_end = N - NUM_GHOSTS

    # Vectorized flux divergence
    rhs_D[i_start:i_end] = -(F_D_face[i_start:i_end] - F_D_face[i_start-1:i_end-1]) * inv_dr
    rhs_S[i_start:i_end, :] = -(F_S_face[i_start:i_end, :] - F_S_face[i_start-1:i_end-1, :]) * inv_dr
    rhs_tau[i_start:i_end] = -(F_tau_face[i_start:i_end] - F_tau_face[i_start-1:i_end-1]) * inv_dr

    return rhs_D, rhs_S, rhs_tau


def extract_valencia_geometry(r: np.ndarray,
                             bssn_vars,
                             spacetime_mode: str,
                             background,
                             grid) -> ValenciaGeometry:
    """
    Extract geometric quantities from BSSN variables.

    Args:
        r: Radial coordinate array
        bssn_vars: BSSN state variables
        spacetime_mode: 'fixed' or 'dynamic'
        background: Background metric object
        grid: Grid object with spacing information

    Returns:
        ValenciaGeometry object containing all geometric quantities
    """
    # Extract BSSN variables
    alpha = np.asarray(bssn_vars.alpha, dtype=float)
    betaU = np.asarray(bssn_vars.betaU, dtype=float)
    phi = np.asarray(bssn_vars.phi, dtype=float)

    # Compute derived quantities
    e6phi = np.exp(6.0 * phi)

    if spacetime_mode == 'fixed':
        gamma_LL = background.gamma_LL
        gamma_UU = background.gamma_UU
        sqrt_gamma = background.sqrt_gamma
        sqrt_g_hat_cell = np.asarray(background.sqrt_g_hat_cell, dtype=float)
    else:  # dynamic
        bar_gamma_LL = get_bar_gamma_LL(bssn_vars.hDD)
        bar_gamma_UU = get_bar_gamma_UU(bssn_vars.hDD)
        det_bar_gamma = get_det_bar_gamma(bar_gamma_LL)
        sqrt_bar_gamma = np.sqrt(det_bar_gamma)

        gamma_LL = e6phi[:, None, None] * bar_gamma_LL
        gamma_UU = np.exp(-6.0 * phi[:, None, None]) * bar_gamma_UU
        sqrt_gamma = e6phi * sqrt_bar_gamma
        sqrt_g_hat_cell = np.asarray(background.sqrt_g_hat_cell, dtype=float)

    dr = grid.dx

    return ValenciaGeometry(
        alpha=alpha,
        beta_U=betaU,
        gamma_LL=gamma_LL,
        gamma_UU=gamma_UU,
        sqrt_gamma=sqrt_gamma,
        e6phi=e6phi,
        sqrt_g_hat_cell=sqrt_g_hat_cell,
        dr=dr
    )


def compute_connection_terms(rho0: np.ndarray,
                            v_U: np.ndarray,
                            pressure: np.ndarray,
                            W: np.ndarray,
                            h: np.ndarray,
                            geometry: ValenciaGeometry,
                            hat_chris: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Christoffel connection contributions to RHS.

    Connection form ( +):
        D, τ:   -Γ̂^k_{kr} F̃^r
        S_i:    -Γ̂^k_{kr} F̃^r_i + Γ̂^l_{ri} F̃^r_l

    Args:
        rho0: Rest mass density
        v_U: Three-velocity (N, 3)
        pressure: Pressure
        W: Lorentz factor
        h: Specific enthalpy
        geometry: Geometric quantities container
        hat_chris: Reference Christoffel symbols (N, 3, 3, 3)

    Returns:
        conn_D: Connection term for density equation
        conn_S: Connection term for momentum equation (N, 3)
        conn_tau: Connection term for energy equation
    """
    N = len(rho0)
    alpha = geometry.alpha
    e6phi = geometry.e6phi
    gamma_LL = geometry.gamma_LL
    gamma_UU = geometry.gamma_UU

    # Compute 4-velocity components
    u4U = np.zeros((N, 4))
    u4U[:, 0] = W / alpha  # u^0
    for i in range(SPACEDIM):
        u4U[:, i+1] = W * v_U[:, i] - geometry.beta_U[:, i] * u4U[:, 0]  # u^i

    # Compute stress-energy tensor T^μν
    u4D = np.zeros((N, 4))
    u4D[:, 0] = -alpha * u4U[:, 0]  # u_0
    for i in range(SPACEDIM):
        for j in range(SPACEDIM):
            u4D[:, i+1] += gamma_LL[:, i, j] * u4U[:, j+1]  # u_i

    # T^μν = ρ₀ h W² g^μν + ρ₀ h W² u^μ u^ν / (1 + u^μ u_μ) + p g^μν
    # For perfect fluid: u^μ u_μ = -1, so factor = ρ₀ h W²
    factor = rho0 * h * W**2

    # Conservative density flux
    rho_star = alpha * e6phi * rho0 * u4U[:, 0]

    # Physical fluxes at cell centers (partial flux vectors)
    fD_U = np.zeros((N, SPACEDIM))
    for j in range(SPACEDIM):
        fD_U[:, j] = rho_star * v_U[:, j]

    # Energy (tau) partial flux vector
    # Compute T^0j first
    T0j = np.zeros((N, SPACEDIM))
    for j in range(SPACEDIM):
        T0j[:, j] = factor * u4U[:, 0] * u4U[:, j+1]

    fTau_U = np.zeros((N, SPACEDIM))
    for j in range(SPACEDIM):
        fTau_U[:, j] = alpha ** 2 * e6phi * T0j[:, j] - rho_star * v_U[:, j]

    # Momentum partial flux tensor F̃^j_i = α e^{6φ} T^j_i
    # First compute T^ij
    Tij_UU = np.zeros((N, SPACEDIM, SPACEDIM))
    for i in range(SPACEDIM):
        for j in range(SPACEDIM):
            Tij_UU[:, i, j] = (factor * u4U[:, i+1] * u4U[:, j+1] +
                               pressure * gamma_UU[:, i, j])

    # Convert to mixed T^j_i
    Tij_UD = np.zeros((N, SPACEDIM, SPACEDIM))
    for j in range(SPACEDIM):
        for i in range(SPACEDIM):
            for k in range(SPACEDIM):
                Tij_UD[:, j, i] += Tij_UU[:, j, k] * gamma_LL[:, i, k]

    F_S_no_ghat = np.zeros((N, SPACEDIM, SPACEDIM))
    for j in range(SPACEDIM):
        for i in range(SPACEDIM):
            F_S_no_ghat[:, j, i] = alpha * e6phi * Tij_UD[:, j, i]

    # Connection contributions
    Gamma_trace = np.einsum('xkkj->xj', hat_chris)
    conn_D = -np.einsum('xj,xj->x', Gamma_trace, fD_U)
    conn_tau = -np.einsum('xj,xj->x', Gamma_trace, fTau_U)
    conn_S = (
        -np.einsum('xl,xli->xi', Gamma_trace, F_S_no_ghat)
        + np.einsum('xlji,xjl->xi', hat_chris, F_S_no_ghat)
    )

    return conn_D, conn_S, conn_tau


def reconstruct_and_convert_valencia(rho0: np.ndarray,
                                    v_primary: np.ndarray,
                                    pressure: np.ndarray,
                                    r: np.ndarray,
                                    reconstructor,
                                    gamma_primary_primary_f: np.ndarray,
                                    eos,
                                    recon_boundary: str,
                                    atmosphere=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct primitive variables and convert to conservatives at cell faces.

    This function handles:
    1. Reconstruction of primitives to cell faces
    2. Boundary condition enforcement
    3. Physical limiters
    4. Conversion to conservative variables

    Args:
        rho0: Rest mass density at cell centers
        v_primary: Primary velocity component (v^r in spherical)
        pressure: Pressure at cell centers
        r: Radial coordinate array
        reconstructor: Reconstruction method object
        gamma_primary_primary_f: Metric component at faces (γ_rr)
        eos: Equation of state
        recon_boundary: Boundary condition type ('reflecting' or 'open')
        atmosphere: Atmosphere handler (optional)

    Returns:
        UL_batch: Left state conservatives (N_faces, 3) [D, S, tau]
        UR_batch: Right state conservatives (N_faces, 3)
        primL_batch: Left state primitives (N_faces, 3) [rho, v, p]
        primR_batch: Right state primitives (N_faces, 3)
    """
    # Reconstruct primitives to cell interfaces
    (rhoL, vL, pL), (rhoR, vR, pR) = reconstructor.reconstruct_primitive_variables(
        rho0, v_primary, pressure, x=r, boundary_type=recon_boundary
    )

    # Extract interior faces (exclude ghost zones)
    rhoL, vL, pL = rhoL[1:-1], vL[1:-1], pL[1:-1]
    rhoR, vR, pR = rhoR[1:-1], vR[1:-1], pR[1:-1]

    # Enforce reflecting condition at the r≈0 interface
    if recon_boundary == "reflecting" and (len(vL) > 0):
        # Enforce v=0 at the innermost face
        vL[0] = 0.0
        vR[0] = 0.0
        # For density and pressure, use symmetric values
        if len(rho0) > NUM_GHOSTS:
            # Use first interior value for consistency
            rho_ref = rho0[NUM_GHOSTS]
            p_ref = pressure[NUM_GHOSTS]
            rhoL[0] = rho_ref
            rhoR[0] = rho_ref
            pL[0] = p_ref
            pR[0] = p_ref

    # Apply physical limiters if available
    if hasattr(reconstructor, "apply_physical_limiters") and atmosphere is not None:
        (rhoL, vL, pL), (rhoR, vR, pR) = reconstructor.apply_physical_limiters(
            (rhoL, vL, pL), (rhoR, vR, pR),
            atmosphere=atmosphere,
            gamma_rr=gamma_primary_primary_f
        )

    # Convert primitives to conservatives at interfaces
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

    return UL_batch, UR_batch, primL_batch, primR_batch


def solve_riemann_and_densitize(UL_batch: np.ndarray,
                               UR_batch: np.ndarray,
                               primL_batch: np.ndarray,
                               primR_batch: np.ndarray,
                               alpha_f: np.ndarray,
                               beta_primary_f: np.ndarray,
                               gamma_primary_primary_f: np.ndarray,
                               e6phi_f: np.ndarray,
                               riemann_solver,
                               eos,
                               boundary_mode: str,
                               spacetime_mode: str,
                               atmosphere=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve Riemann problem and apply densitization.

    Args:
        UL_batch, UR_batch: Conservative states at faces
        primL_batch, primR_batch: Primitive states at faces
        alpha_f, beta_primary_f: Lapse and shift at faces
        gamma_primary_primary_f: Metric component at faces
        e6phi_f: exp(6φ) at faces
        riemann_solver: Riemann solver object
        eos: Equation of state
        boundary_mode: 'parity' or other
        spacetime_mode: 'fixed_minkowski', 'fixed', or 'dynamic'
        atmosphere: Atmosphere handler (optional)

    Returns:
        F_D_face: Density flux at faces
        F_S_tildeD_face: Momentum flux at faces (3D)
        F_tau_face: Energy flux at faces
    """
    # Solve Riemann problem to get physical fluxes
    F_phys_batch = riemann_solver.solve_batch(
        UL_batch, UR_batch, primL_batch, primR_batch,
        gamma_primary_primary_f, alpha_f, beta_primary_f, eos
    )

    # Densitization strictly α e^{6φ} at faces
    dens_factor = alpha_f * e6phi_f
    F_batch = dens_factor[:, None] * F_phys_batch

    # Enforce zero momentum flux at the r≈0 interface (spherical boundary)
    if boundary_mode == "parity" and spacetime_mode != "fixed_minkowski":
        if len(F_batch) > 0:
            F_batch[0, 1] = 0.0  # F_Sr = 0 at the r≈0 interface

    # Additional freeze: zero fluxes at atmosphere interfaces
    if atmosphere is not None and spacetime_mode != "fixed_minkowski":
        try:
            rho_atm_face = atmosphere.rho_floor * getattr(atmosphere, 'rho_threshold', 1.0)
            rhoL = primL_batch[:, 0]
            rhoR = primR_batch[:, 0]
            mask_freeze = (rhoL < rho_atm_face) | (rhoR < rho_atm_face)
            if np.any(mask_freeze):
                F_batch[mask_freeze, :] = 0.0
        except Exception:
            pass

    # Construct 3D momentum flux array
    N_faces = len(F_batch)
    F_S_tildeD_face = np.zeros((N_faces, SPACEDIM))
    F_S_tildeD_face[:, 0] = F_batch[:, 1]  # Primary direction flux

    return F_batch[:, 0], F_S_tildeD_face, F_batch[:, 2]


def compute_source_terms_valencia(rho0: np.ndarray,
                                 v_U: np.ndarray,
                                 pressure: np.ndarray,
                                 W: np.ndarray,
                                 h: np.ndarray,
                                 geometry: ValenciaGeometry,
                                 bssn_vars,
                                 bssn_d1,
                                 background,
                                 spacetime_mode: str,
                                 r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute geometric source terms for Valencia equations.

    Source terms arise from:
    1. Extrinsic curvature K_ij contractions
    2. Lapse gradient ∂_i α
    3. Shift covariant derivative ∇̂_i β^j
    4. Metric covariant derivative ∇̂_i γ_{jk}

    Args:
        rho0: Rest mass density
        v_U: Three-velocity (N, 3)
        pressure: Pressure
        W: Lorentz factor
        h: Specific enthalpy
        geometry: Geometric quantities container
        bssn_vars: BSSN variables
        bssn_d1: BSSN derivatives
        background: Background metric
        spacetime_mode: 'fixed' or 'dynamic'
        r: Radial coordinate

    Returns:
        src_S: Momentum source terms (N, 3)
        src_tau: Energy source term (N,)
    """
    N = len(rho0)

    # Fixed spacetime has no source terms
    if spacetime_mode == 'fixed':
        src_S_vector = np.zeros((N, SPACEDIM))
        src_tau = np.zeros(N)
        return src_S_vector, src_tau

    # Extract needed quantities
    alpha = geometry.alpha
    beta_U = geometry.beta_U
    gamma_LL = geometry.gamma_LL
    gamma_UU = geometry.gamma_UU
    e6phi = geometry.e6phi

    phi = np.asarray(bssn_vars.phi, dtype=float)
    e4phi = np.exp(4.0 * phi)

    # Compute 4-velocity and stress-energy tensor
    u4U = np.zeros((N, 4))
    u4U[:, 0] = W / alpha  # u^0
    for i in range(SPACEDIM):
        u4U[:, i+1] = W * v_U[:, i] - beta_U[:, i] * u4U[:, 0]  # u^i

    # Compute T^μν (contravariant)
    u4D = np.zeros((N, 4))
    u4D[:, 0] = -alpha * u4U[:, 0]  # u_0
    for i in range(SPACEDIM):
        for j in range(SPACEDIM):
            u4D[:, i+1] += gamma_LL[:, i, j] * u4U[:, j+1]  # u_i

    factor = rho0 * h * W**2

    # T^00
    T00 = factor * u4U[:, 0]**2 + pressure

    # T^0i
    T0i = np.zeros((N, SPACEDIM))
    for i in range(SPACEDIM):
        T0i[:, i] = factor * u4U[:, 0] * u4U[:, i+1]

    # T^ij
    Tij = np.zeros((N, SPACEDIM, SPACEDIM))
    for i in range(SPACEDIM):
        for j in range(SPACEDIM):
            Tij[:, i, j] = (factor * u4U[:, i+1] * u4U[:, j+1] +
                           pressure * gamma_UU[:, i, j])

    # T^0_i (mixed)
    T0i_UD = np.zeros((N, SPACEDIM))
    for i in range(SPACEDIM):
        for j in range(SPACEDIM):
            T0i_UD[:, i] += T0i[:, j] * gamma_LL[:, i, j]

    # ====================================================================
    # ENERGY SOURCE TERM
    # ====================================================================

    # Extrinsic curvature K_ij = e^{4φ} Ā_ij + (K/3) γ_ij
    K = np.asarray(bssn_vars.K, dtype=float)
    bar_A_LL = get_bar_A_LL(bssn_vars.aDD)
    bar_gamma_LL = get_bar_gamma_LL(bssn_vars.hDD)
    K_LL = e4phi[:, None, None] * bar_A_LL + (K / 3.0)[:, None, None] * gamma_LL

    # Lapse derivatives ∂_i α
    dalpha_dx = np.zeros((N, SPACEDIM))
    if hasattr(bssn_d1, 'lapse') and bssn_d1.lapse is not None:
        dalpha_dx = np.asarray(bssn_d1.lapse)

    # Term 1: K_ij contraction
    tensor_block_energy = (
        T00[:, None, None] * np.einsum('xi,xj->xij', beta_U, beta_U)
        + 2.0 * np.einsum('xi,xj->xij', T0i, beta_U)
        + Tij
    )
    term1_tau = np.einsum('xij,xij->x', K_LL, tensor_block_energy)

    # Term 2: lapse derivative term
    term2_tau = -(
        np.einsum('x,xi,xi->x', T00, beta_U, dalpha_dx)
        + np.einsum('xi,xi->x', T0i, dalpha_dx)
    )

    # Final energy source term
    src_tau = alpha * e6phi * (term1_tau + term2_tau)

    # ====================================================================
    # MOMENTUM SOURCE TERM
    # ====================================================================

    # Shift derivatives ∂_i β^j
    dbeta_dx = np.zeros((N, SPACEDIM, SPACEDIM))
    if hasattr(bssn_d1, 'shift_U') and bssn_d1.shift_U is not None:
        shift_d1 = np.asarray(bssn_d1.shift_U)
        if shift_d1.ndim >= 3:
            for i in range(SPACEDIM):
                for j in range(SPACEDIM):
                    dbeta_dx[:, i, j] = (
                        background.inverse_scaling_vector[:, i] * shift_d1[:, i, j]
                        + bssn_vars.shift_U[:, i] * background.d1_inverse_scaling_vector[:, i, j]
                    )

    # Reference metric Christoffel symbols Γ̂^i_{jk}
    hat_chris = background.hat_christoffel

    # Covariant derivative of shift: ∇̂_i β^j = ∂_i β^j + Γ̂^j_{ik} β^k
    hatD_beta_U = dbeta_dx + np.einsum('xjik,xk->xij', hat_chris, beta_U)

    # Covariant derivative of metric: ∇̂_i γ_{jk}
    dphi_dx = np.zeros((N, SPACEDIM))
    if hasattr(bssn_d1, 'phi') and bssn_d1.phi is not None:
        dphi_dx = np.asarray(bssn_d1.phi)

    hat_D_bar_gamma = get_hat_D_bar_gamma_LL(bssn_vars.hDD, bssn_d1.hDD)

    # ∇̂_i γ_{jk} = e^{4φ} [4 γ̄_{jk} ∂_i φ + ∇̂_i γ̄_{jk}]
    hatD_gamma_LL = np.zeros((N, SPACEDIM, SPACEDIM, SPACEDIM))
    for i in range(SPACEDIM):
        for j in range(SPACEDIM):
            for k in range(SPACEDIM):
                hatD_gamma_LL[:, i, j, k] = (
                    4.0 * bar_gamma_LL[:, j, k] * dphi_dx[:, i]
                    + hat_D_bar_gamma[:, j, k, i]
                )
                hatD_gamma_LL[:, i, j, k] *= e4phi

    # Term 1: -T^{00} α ∂_i α
    first_term = -T00[:, None] * alpha[:, None] * dalpha_dx

    # Term 2: T^0_j ∇̂_i β^j
    second_term = np.einsum('xj,xij->xi', T0i_UD, hatD_beta_U)

    # Term 3: (1/2) (T^{00} β^j β^k + 2 T^{0j} β^k + T^{jk}) ∇̂_i γ_{jk}
    tensor_block_momentum = (
        T00[:, None, None] * np.einsum('xj,xk->xjk', beta_U, beta_U)
        + 2.0 * np.einsum('xj,xk->xjk', T0i, beta_U)
        + Tij
    )
    third_term = 0.5 * np.einsum('xjk,xijk->xi', tensor_block_momentum, hatD_gamma_LL)

    # Combine with volume element
    src_S_vector = alpha[:, None] * e6phi[:, None] * (
        first_term + second_term + third_term
    )

    return src_S_vector, src_tau  # (N, 3), (N,)


def dUdt_valencia(D: np.ndarray,
                 S_tildeD: np.ndarray,
                 tau: np.ndarray,
                 rho0: np.ndarray,
                 v_U: np.ndarray,
                 pressure: np.ndarray,
                 W: np.ndarray,
                 h: np.ndarray,
                 r: np.ndarray,
                 bssn_vars,
                 bssn_d1,
                 background,
                 spacetime_mode: str,
                 eos,
                 grid,
                 reconstructor,
                 riemann_solver,
                 atmosphere,
                 boundary_mode: str = 'reflecting') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute RHS of GRHD equations.

    This function implements the finite volume scheme:
    1. Extract geometry
    2. Reconstruct primitives
    3. Solve Riemann problem
    4. Compute flux divergence
    5. Add connection terms
    6. Add source terms

    Args:
        D, S_tildeD, tau: Conservative variables
        rho0, v_U, pressure, W, h: Primitive variables and derived quantities
        r: Radial coordinate
        bssn_vars, bssn_d1: BSSN variables and derivatives
        background: Background metric
        spacetime_mode: 'fixed' or 'dynamic'
        eos: Equation of state
        grid: Grid object
        reconstructor: Reconstruction method
        riemann_solver: Riemann solver
        atmosphere: Atmosphere handler
        boundary_mode: Boundary condition type

    Returns:
        rhs_D, rhs_S_tildeD, rhs_tau: RHS of evolution equations
    """
    # 1. Extract geometry
    geometry = extract_valencia_geometry(r, bssn_vars, spacetime_mode, background, grid)

    # Prepare geometry at faces (interpolation)
    phi_arr = np.asarray(bssn_vars.phi, dtype=float)
    alpha_f = 0.5 * (geometry.alpha[:-1] + geometry.alpha[1:])
    beta_U_f = 0.5 * (geometry.beta_U[:-1] + geometry.beta_U[1:])
    gamma_LL_f = 0.5 * (geometry.gamma_LL[:-1] + geometry.gamma_LL[1:])
    phi_face = 0.5 * (phi_arr[:-1] + phi_arr[1:])
    e6phi_f = np.exp(6.0 * phi_face)

    # Extract primary spatial components for 1D Riemann solver
    v_primary = v_U[:, 0]  # v^r in spherical
    beta_primary_f = beta_U_f[:, 0]
    gamma_primary_primary_f = gamma_LL_f[:, 0, 0]

    # Determine reconstruction boundary type
    recon_boundary = "reflecting" if boundary_mode == 'reflecting' else "open"

    # 2. Reconstruct primitives and convert to conservatives
    UL_batch, UR_batch, primL_batch, primR_batch = reconstruct_and_convert_valencia(
        rho0, v_primary, pressure, r, reconstructor,
        gamma_primary_primary_f, eos, recon_boundary, atmosphere
    )

    # 3. Solve Riemann problem and densitize
    F_D_face, F_S_face, F_tau_face = solve_riemann_and_densitize(
        UL_batch, UR_batch, primL_batch, primR_batch,
        alpha_f, beta_primary_f, gamma_primary_primary_f, e6phi_f,
        riemann_solver, eos, "parity" if boundary_mode == 'reflecting' else "open",
        spacetime_mode, atmosphere
    )

    # 4. Compute flux divergence 
    rhs_D, rhs_S, rhs_tau = compute_divergence_vectorized(
        F_D_face, F_S_face, F_tau_face, geometry.dr
    )

    # 5. Add connection terms
    hat_chris = background.hat_christoffel
    conn_D, conn_S, conn_tau = compute_connection_terms(
        rho0, v_U, pressure, W, h, geometry, hat_chris
    )
    rhs_D += conn_D
    rhs_S += conn_S
    rhs_tau += conn_tau

    # 6. Add source terms
    src_S, src_tau = compute_source_terms_valencia(
        rho0, v_U, pressure, W, h, geometry, bssn_vars, bssn_d1,
        background, spacetime_mode, r
    )
    rhs_S += src_S
    rhs_tau += src_tau

    return rhs_D, rhs_S, rhs_tau

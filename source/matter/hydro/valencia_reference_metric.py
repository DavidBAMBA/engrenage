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
                 atmosphere_rho=None, p_floor=None, v_max=None,
                 enable_surface_flattening=True,
                 rho_surface_threshold=1.0e-6,
                 surface_jump_max=50.0):
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

        # Controls for added robustness at the stellar surface
        self.enable_surface_flattening = bool(enable_surface_flattening)
        self.rho_surface_threshold = float(rho_surface_threshold)
        self.surface_jump_max = float(surface_jump_max)

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
            # The factor √(det γ̂) is handled implicitly via Γ̂^i_{jk}
            self.sqrt_gamma = e6phi
            self.e6phi = e6phi  # Store for later use

            # √ĝ for reference metric
            self.sqrt_g_hat_cell = np.sqrt(np.abs(background.det_hat_gamma) + 1e-30)
            self.sqrt_g_hat_face = 0.5 * (self.sqrt_g_hat_cell[:-1] + self.sqrt_g_hat_cell[1:])

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

    def _compute_T4UD(self, T4UU):
        """
        Compute mixed stress-energy tensor T^μ_ν = T^{μδ} g_{δν} following NRPy+ (line 176-198).

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

    def _compute_source_terms(self, rho0, v_U, pressure, W, h,
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

        # Geometric quantities from class attributes
        alpha = self.alpha
        beta_U = self.beta_U  # (N, SPACEDIM) - all three components
        
        # Compute e^{6φ} and √(det γ̄) separately for clarity
        # Use only e^{6φ} as volume element (GRoovy Eq. 20)
        # The √(det γ̂) factor is implicitly handled via Γ̂^i_{jk}
        phi = np.asarray(bssn_vars.phi, dtype=float)
        e6phi = np.exp(6.0 * phi)
        vol_element = e6phi

        
        # Build full 3D metric γ_ij = e^{4φ} γ̄_ij
        e4phi = np.exp(4.0 * phi)
        bar_gamma_LL = get_bar_gamma_LL(r, bssn_vars.h_LL, background)
        gamma_LL = e4phi[:, None, None] * bar_gamma_LL
        
        # Compute stress-energy tensors T^{μν} and T^μ_ν with all three velocity components
        T4UU = self._compute_T4UU(rho0, v_U, pressure, W, h)
        T4UD = self._compute_T4UD(T4UU)
        
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
        # (NRPy+ line 422-423)
        hatD_beta_U = dbeta_dx + np.einsum('xjik,xk->xij', hat_chris, beta_U)

        # Covariant derivative of metric: ∇̂_i γ_{jk} (NRPy+ line 407-415)
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
        
        # --- Debug: print once contributions at first interior cell (r ~ 0) ---
        if not getattr(self, '_debug_source_once', False):
            try:
                i0 = NUM_GHOSTS
                print("\n[Valencia debug] Source-term internals (first interior cell):")
                print(f"  r[i0]={r[i0]:.6e}, alpha={alpha[i0]:.6e}, e6phi={vol_element[i0]:.6e} (≈√γ)")
                # Angular velocities should be zero by spherical symmetry
                print(f"  v_theta={v_U[i0,1]:.6e}, v_phi={v_U[i0,2]:.6e}")
                # T^{θθ}, T^{φφ}
                print(f"  T4UU[θθ]={T4UU['ij'][i0,1,1]:.6e}, T4UU[φφ]={T4UU['ij'][i0,2,2]:.6e}")
                # ∇̂_r γ_{θθ}, ∇̂_r γ_{φφ}
                print(f"  hatD_gamma_LL[r,θ,θ]={hatD_gamma_LL[i0, i_r, i_t, i_t]:.6e}, hatD_gamma_LL[r,φ,φ]={hatD_gamma_LL[i0, i_r, i_p, i_p]:.6e}")
                # ∂_r α
                print(f"  dalpha_dr={dalpha_dx[i0, i_r]:.6e}")
                # Momentum source terms components
                print(f"  first_term[r]={first_term[i0]:.6e}, second_term[r]={second_term[i0]:.6e}, third_term[r]={third_term[i0]:.6e}")
                print(f"  src_S_vector[r]={src_S_vector[i0,0]:.6e}")
            except Exception:
                pass
            self._debug_source_once = True

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

        # Compute NRPy-style partial fluxes at interfaces using α e^{6φ} at faces
        F_D_face, F_S_tildeD_face, F_tau_face, debug_faces = self._compute_interface_fluxes(
            rho0, v_U, pressure, r, eos, reconstructor, riemann_solver, bssn_vars
        )

        # Reference metric Christoffel symbols Γ̂^i_{jk}
        hat_chris = background.hat_christoffel  # (N, 3, 3, 3)

        # ====================================================================
        # FLUX DIVERGENCE (NRPy form): -∂_r(F̃^r)
        #   with F̃^r = α √γ F_phys^r (no √ĝ factor here)
        # ====================================================================

        rhs_D = np.zeros(N)
        rhs_S_tildeD = np.zeros((N, SPACEDIM))  
        rhs_tau = np.zeros(N)

        # Compute divergence in interior cells
        for i in range(NUM_GHOSTS, N - NUM_GHOSTS):
            # NRPy-style partial derivative form: divide only by Δr
            inv_vol = 1.0 / (self.dr + 1e-30)

            rhs_D[i] = -(F_D_face[i] - F_D_face[i-1]) * inv_vol
            
            # All three momentum components
            rhs_S_tildeD[i, :] = -(F_S_tildeD_face[i, :] - F_S_tildeD_face[i-1, :]) * inv_vol
            
            rhs_tau[i] = -(F_tau_face[i] - F_tau_face[i-1]) * inv_vol

        # ====================================================================
        # CONNECTION TERMS (NRPy form):
        #   D, τ:      -Γ̂^k_{kr} F̃^r
        #   S_i:       -Γ̂^k_{kr} F̃^r_i + Γ̂^l_{ri} F̃^r_l
        # ====================================================================

        # Use e^{6φ} (not √γ) for densitization at centers, matching NRPy.
        alpha = self.alpha
        phi_c = np.asarray(bssn_vars.phi, dtype=float)
        e6phi_c = np.exp(6.0 * phi_c)


        # Compute 4-velocity and stress-energy
        u4U = self._compute_4velocity(rho0, v_U, W)
        T4UU = self._compute_T4UU(rho0, v_U, pressure, W, h)

        # Conservative density
        rho_star = alpha * e6phi_c * rho0 * u4U[:, 0]

        # Physical fluxes at cell centers
        fD_U = np.zeros((N, SPACEDIM))
        for j in range(SPACEDIM):
            fD_U[:, j] = rho_star * v_U[:, j]

        # Energy (tau) partial flux vector, densitized:
        #   F̃^j_tau = α e^{6φ} ( α T^{0j} ) - ρ_* v^j
        fTau_U = np.zeros((N, SPACEDIM))
        for j in range(SPACEDIM):
            fTau_U[:, j] = alpha ** 2 * e6phi_c * T4UU['0i'][:, j] - rho_star * v_U[:, j]

        # Momentum partial flux tensor F̃^j_i = α e^{6φ} T^j_i
        T4UD = self._compute_T4UD(T4UU)
        F_S_no_ghat = np.zeros((N, SPACEDIM, SPACEDIM))
        for j in range(SPACEDIM):
            for i in range(SPACEDIM):
                F_S_no_ghat[:, j, i] = alpha * e6phi_c * T4UD['i_j'][:, j, i]

        # Debug print once: inspect first-interior-cell flux balance near origin
        if not getattr(self, '_debug_printed_once', False):
            try:
                i0 = NUM_GHOSTS
                kL = i0 - 1
                kR = i0
                print("\n[Valencia debug] First interior cell diagnostics:")
                print(f"  i0={i0}, r={r[i0]:.6e}")
                # Face fluxes (densitized) entering divergence
                print(f"  Faces kL={kL}, kR={kR} (len faces={len(F_D_face)})")
                print(f"    F_D_face[kL:kR]    = {[F_D_face[kL], F_D_face[kR]] if kR < len(F_D_face) else 'n/a'}")
                print(f"    F_tau_face[kL:kR]  = {[F_tau_face[kL], F_tau_face[kR]] if kR < len(F_tau_face) else 'n/a'}")
                print(f"    F_S_face_r[kL:kR]  = {[F_S_tildeD_face[kL,0], F_S_tildeD_face[kR,0]] if kR < len(F_S_tildeD_face) else 'n/a'}")
                # Center flux pieces
                print(f"  Center densitization: alpha={alpha[i0]:.6e}, e6phi={e6phi_c[i0]:.6e}")
                print(f"    fD_U[i0,0]   = {fD_U[i0,0]:.6e}")
                print(f"    fTau_U[i0,0] = {fTau_U[i0,0]:.6e}")
                print(f"    F_S_c_r[i0]  = {F_S_no_ghat[i0,0,0]:.6e}")
                # Connection pieces
                Gamma_trace = np.einsum('xkkj->xj', hat_chris)
                print(f"  Gamma_trace_r[i0] = {Gamma_trace[i0,0]:.6e}")
                # Face densitization params if available
                if debug_faces:
                    af = debug_faces.get('alpha_f', None)
                    e6f = debug_faces.get('e6phi_f', None)
                    rhoL = debug_faces.get('rhoL', None)
                    rhoR = debug_faces.get('rhoR', None)
                    vL = debug_faces.get('vL', None)
                    vR = debug_faces.get('vR', None)
                    pL = debug_faces.get('pL', None)
                    pR = debug_faces.get('pR', None)
                    if af is not None and e6f is not None and kR < len(af):
                        print(f"  Face params: alpha_f[kL:kR]={[af[kL], af[kR]] if kR < len(af) else 'n/a'}, e6phi_f[kL:kR]={[e6f[kL], e6f[kR]] if kR < len(e6f) else 'n/a'}")
                    if rhoL is not None and vL is not None and pL is not None:
                        print(f"  Reconstructed at kL: rhoL={rhoL[kL]:.6e}, vL={vL[kL]:.6e}, pL={pL[kL]:.6e}")
                    if rhoR is not None and vR is not None and pR is not None:
                        print(f"  Reconstructed at kL: rhoR={rhoR[kL]:.6e}, vR={vR[kL]:.6e}, pR={pR[kL]:.6e}")
                self._debug_printed_once = True
            except Exception as _e:
                # Don't let debugging break evolution
                self._debug_printed_once = True

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
            rho0, v_U, pressure, W, h, bssn_vars, bssn_d1,
            background, spacetime_mode, r
        )  # Returns (N,3), (N,)

        rhs_S_tildeD += src_S_tildeD  # Vector addition - all three components
        rhs_tau += src_tau

        # Second debug block: full balance in first interior cell
        if not getattr(self, '_debug_balance_printed', False):
            try:
                i0 = NUM_GHOSTS
                kL = i0 - 1
                kR = i0
                inv_vol = 1.0 / (self.dr + 1e-30)
                divS_i0 = -(F_S_tildeD_face[kR, 0] - F_S_tildeD_face[kL, 0]) * inv_vol
                # Connection term at center for i=radial
                Gamma_trace = np.einsum('xkkj->xj', hat_chris)
                term1 = -np.einsum('j,ji->', Gamma_trace[i0, :], F_S_no_ghat[i0, :, :][:, 0])
                term2 = np.einsum('lji,jl->', hat_chris[i0, :, :, 0], F_S_no_ghat[i0, :, :])
                connS_i0 = term1 + term2
                srcS_i0 = src_S_tildeD[i0, 0]
                print("[Valencia debug] Balance at first interior cell:")
                print(f"  divS = {divS_i0:.6e}, connS = {connS_i0:.6e}, srcS = {srcS_i0:.6e}")
                print(f"  rhs_S(i0) = {rhs_S_tildeD[i0,0]:.6e}")
                self._debug_balance_printed = True
            except Exception:
                self._debug_balance_printed = True

        # Preserve backward-compatible shape: if input momentum was 1D, return RHS radial as (N,)
        if _return_radial_only:
            rhs_S_out = rhs_S_tildeD[:, 0]
        else:
            rhs_S_out = rhs_S_tildeD

        return rhs_D, rhs_S_out, rhs_tau  # (N,), (N,) or (N,3), (N,)

    def _compute_interface_fluxes(self, rho0, v_U, pressure, r,
                                eos, reconstructor, riemann_solver, bssn_vars):
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
            Tuple of partial-flux arrays at interfaces (no √ĝ factor):
                F_D_face: (N_faces,) density flux
                F_S_tildeD_face: (N_faces, 3) momentum flux (only [:, 0] non-zero)
                F_tau_face: (N_faces,) energy flux
                debug_info: dict with diagnostic information
        """
        N = len(r)

        # Extract first spatial component for 1D reconstruction
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

        # Interpolate geometry to faces from class attributes
        alpha_f = 0.5 * (self.alpha[:-1] + self.alpha[1:])
        sqrt_gamma_f = 0.5 * (self.sqrt_gamma[:-1] + self.sqrt_gamma[1:])
        sqrt_g_hat_f = self.sqrt_g_hat_face
        # Exact e^{6phi} at faces from BSSN phi (NRPy-compatible densitization)
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
        
        # ========== COMMENTED OUT:  ==========
        # This "atmosphere flattening" code averages reconstructed states near
        # low-density regions. However:
        # 1. NRPy+ does NOT implement this (confirmed via codebase search)
        # 2. Not mentioned in GRHD literature (Font, Baumgarte, etc.)
        # 3. Destroys high-order reconstruction (WENO5 → order-0 averaging)
        # 4. The diagnostic shows WENO5 produces negative density overshoots
        #    at the sharp surface discontinuity (rhoL = -4e-7), but flattening
        #    hides this rather than fixing the root cause
        #
        # The correct approach is to use apply_physical_limiters() in
        # reconstruction.py which applies floors WITHOUT destroying high-order
        # reconstruction everywhere.
        #
        # Original code:
        # dens_floor = max(self.atmosphere.rho_floor, 1e-20)
        # thr = 30.0 * dens_floor
        # if len(rhoL) > 0:
        #     mask = (rhoL < thr) | (rhoR < thr)
        #     if np.any(mask):
        #         rho_avg = 0.5 * (rhoL + rhoR)
        #         p_avg = np.maximum(0.5 * (pL + pR), self.atmosphere.p_floor)
        #         rhoL[mask] = rho_avg[mask]
        #         rhoR[mask] = rho_avg[mask]
        #         pL[mask] = p_avg[mask]
        #         pR[mask] = p_avg[mask]
        #         vL[mask] = 0.0
        #         vR[mask] = 0.0
        # ========== END COMMENT ==========

        # ========== ALSO COMMENTED OUT: Surface-aware flattening ==========
        # This is another averaging/flattening strategy that is not in NRPy+.
        # Same issues as above: destroys high-order reconstruction.
        #
        # Original code:
        # if self.enable_surface_flattening and len(rhoL) > 0:
        #     surface_thr = max(self.rho_surface_threshold, 100.0 * self.atmosphere.rho_floor)
        #     denom = np.maximum(np.minimum(rhoL, rhoR), 1e-30)
        #     jump_ratio = np.maximum(rhoL, rhoR) / denom
        #     mask_surf = (rhoL < surface_thr) | (rhoR < surface_thr) | (jump_ratio > 10.0)
        #     if np.any(mask_surf):
        #         extreme_jump = jump_ratio > 1000.0
        #         rho_avg = 0.5 * (rhoL + rhoR)
        #         p_avg = np.maximum(0.5 * (pL + pR), self.atmosphere.p_floor)
        #         rhoL[mask_surf] = np.maximum(rho_avg[mask_surf], self.atmosphere.rho_floor)
        #         rhoR[mask_surf] = np.maximum(rho_avg[mask_surf], self.atmosphere.rho_floor)
        #         pL[mask_surf] = p_avg[mask_surf]
        #         pR[mask_surf] = p_avg[mask_surf]
        #         vL[mask_surf] = 0.0
        #         vR[mask_surf] = 0.0
        #         if np.any(extreme_jump):
        #             rhoL[extreme_jump] = self.atmosphere.rho_floor
        #             rhoR[extreme_jump] = self.atmosphere.rho_floor
        #             pL[extreme_jump] = self.atmosphere.p_floor
        #             pR[extreme_jump] = self.atmosphere.p_floor
        #             vL[extreme_jump] = 0.0
        #             vR[extreme_jump] = 0.0
        # ========== END COMMENT ==========
        
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
        if self.boundary_mode == "parity":
            # The first face must have zero radial momentum flux
            # to enforce the reflecting boundary condition at r=0
            if len(F_batch) > 0:
                F_batch[0, 1] = 0.0  # F_Sr = 0 at the r≈0 interface

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

        # Debug info for diagnostics
        debug_info = {
            'alpha_f': alpha_f,
            'e6phi_f': e6phi_f,
            'rhoL': rhoL, 'rhoR': rhoR,
            'vL': vL, 'vR': vR,
            'pL': pL, 'pR': pR,
        }

        return F_batch[:, 0], F_S_tildeD_face, F_batch[:, 2], debug_info


# valencia_reference_metric.py
"""Valencia formulation with reference metric - full 3D."""

import numpy as np
import time
from contextlib import contextmanager

from source.bssn.tensoralgebra import (
    SPACEDIM,
    get_bar_gamma_LL,
    get_bar_A_LL,
)
from source.bssn.tensoralgebra_kernels import (
    inv_3x3,
    compute_hat_D_bar_gamma_LL as get_hat_D_bar_gamma_LL
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
from source.matter.hydro.valencia_kernels import (
    compute_T4UU_kernel,
    compute_T4UD_kernel,
    compute_source_terms_kernel,
    compute_source_terms_fused_kernel,
    compute_connection_terms_kernel,
    compute_fluxes_kernel,
    compute_hatD_beta_kernel,
    compute_hatD_gamma_kernel
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

        # Performance profiling
        self.enable_profiling = False
        self.timers = {
            'extraction_geometry': 0.0,
            'cons2prim': 0.0,
            'reconstruction': 0.0,
            'riemann_solver': 0.0,
            'source_terms': 0.0,
            'total_rhs': 0.0
        }
        self.call_counts = {k: 0 for k in self.timers.keys()}

    @contextmanager
    def timer(self, name):
        """Context manager for timing code blocks."""
        if not self.enable_profiling:
            yield
            return

        t0 = time.perf_counter()
        yield
        elapsed = time.perf_counter() - t0
        self.timers[name] += elapsed
        self.call_counts[name] += 1

    def print_profiling_stats(self):
        """Print timing statistics."""
        if not self.enable_profiling:
            print("Profiling is disabled. Set valencia.enable_profiling = True")
            return

        print("\n" + "="*60)
        print("VALENCIA PROFILING STATISTICS")
        print("="*60)

        for name in ['extraction_geometry', 'cons2prim', 'reconstruction',
                     'riemann_solver', 'source_terms', 'total_rhs']:
            t = self.timers[name]
            c = self.call_counts[name]
            avg = t / c if c > 0 else 0.0
            pct = 100.0 * t / self.timers['total_rhs'] if self.timers['total_rhs'] > 0 else 0.0

            print(f"{name:20s}: {t:8.3f}s total | {avg*1000:6.2f}ms avg | {pct:5.1f}% | {c:5d} calls")

        print("="*60)

    def reset_profiling_stats(self):
        """Reset all timers."""
        for k in self.timers.keys():
            self.timers[k] = 0.0
            self.call_counts[k] = 0

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
        self.gamma_UU = inv_3x3(self.gamma_LL)

        # Cache for source terms (avoid recomputation)
        self.e4phi = e4phi
        self.e6phi = e6phi
        self.bar_gamma_LL = bar_gamma_LL

        # Pre-compute face-interpolated geometry (avoid recomputation in _compute_interface_fluxes)
        self.alpha_face = 0.5 * (self.alpha[:-1] + self.alpha[1:])
        self.e6phi_face = 0.5 * (e6phi[:-1] + e6phi[1:])
        self.beta_U_face = 0.5 * (self.beta_U[:-1] + self.beta_U[1:])
        self.gamma_LL_face = 0.5 * (self.gamma_LL[:-1] + self.gamma_LL[1:])

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
        N = len(rho0)

        # Allocate output arrays
        T00 = np.empty(N)
        T0i = np.empty((N, SPACEDIM))
        Tij = np.empty((N, SPACEDIM, SPACEDIM))

        # Call Numba kernel
        compute_T4UU_kernel(rho0, v_U, pressure, W, h,
                           self.alpha, self.beta_U, self.gamma_UU,
                           T00, T0i, Tij)

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
        N = len(T00)

        # Allocate output arrays
        T0_0 = np.empty(N)
        T0_j = np.empty((N, SPACEDIM))
        Ti_j = np.empty((N, SPACEDIM, SPACEDIM))

        # Call Numba kernel
        compute_T4UD_kernel(T00, T0i, Tij,
                           self.alpha, self.beta_U, self.gamma_LL,
                           T0_0, T0_j, Ti_j)

        return T0_0, T0_j, Ti_j


#------------------------------------------------------------------------------
# GRHD source, flux, and connection term computations
#------------------------------------------------------------------------------
    def _compute_source_terms(self, rho0, v_U, pressure, W, h,
                            bssn_vars, bssn_d1, background, spacetime_mode, r):
        """
        Physical source terms of GRHD equations in Fully 3D implementation.

        Uses fully fused kernel that computes T^{μν}, T^μ_ν, K_{ij},
        covariant derivatives, and source terms all in one parallel loop.

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

        # Use cached geometry from _extract_geometry (already contiguous)
        alpha = self.alpha
        beta_U = self.beta_U
        e4phi = self.e4phi
        e6phi = self.e6phi
        gamma_LL = self.gamma_LL
        gamma_UU = self.gamma_UU

        # BSSN variables needed for source terms
        K_trace = np.asarray(bssn_vars.K, dtype=float)
        bar_A_LL = np.ascontiguousarray(background.scaling_matrix * bssn_vars.a_LL)

        # Derivatives (np.asarray returns contiguous by default)
        dalpha_dx = np.asarray(bssn_d1.lapse, dtype=float)
        dphi_dx = np.asarray(bssn_d1.phi, dtype=float)

        # Shift derivatives: d(β^i)/dx^j (transposed for kernel)
        dbeta_dx = (
            background.inverse_scaling_vector[:, :, None] * np.asarray(bssn_d1.shift_U)
            + bssn_vars.shift_U[:, :, None] * background.d1_inverse_scaling_vector
        )
        dbeta_dx_transposed = np.ascontiguousarray(np.transpose(dbeta_dx, (0, 2, 1)))

        # Covariant derivative of conformal metric
        hat_D_bar_gamma = get_hat_D_bar_gamma_LL(r, bssn_vars.h_LL, bssn_d1.h_LL, background)

        # Reference metric Christoffel symbols (already contiguous from background)
        hat_chris = background.hat_christoffel

        # Allocate output arrays
        src_S_vector = np.empty((N, SPACEDIM))
        src_tau = np.empty(N)

        # Call fully fused kernel
        compute_source_terms_fused_kernel(
            rho0, v_U, pressure, W, h,
            alpha, beta_U, e4phi, e6phi, gamma_LL, gamma_UU,
            K_trace, bar_A_LL,
            dalpha_dx, dphi_dx, dbeta_dx_transposed, hat_D_bar_gamma,
            hat_chris,
            src_S_vector, src_tau
        )

        return src_S_vector, src_tau


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
            fS_UD: (N, 3, 3) momentum partial flux tensor
        """
        N = len(rho0)

        # Allocate output arrays
        fD_U = np.empty((N, SPACEDIM))
        fTau_U = np.empty((N, SPACEDIM))
        fS_UD = np.empty((N, SPACEDIM, SPACEDIM))

        # Call Numba kernel
        compute_fluxes_kernel(N, rho0, v_U, pressure, W, h,
                             alpha, e6phi, gamma_LL, gamma_UU, beta_U,
                             fD_U, fTau_U, fS_UD)

        return fD_U, fTau_U, fS_UD

    def _compute_connection_terms(self, rho0, v_U, pressure, W, h, background):
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
        fD_U, fTau_U, fS_UD = self._compute_fluxes(
            rho0, v_U, pressure, W, h,
            self.alpha, self.e6phi, self.gamma_LL, self.gamma_UU, self.beta_U
        )

        # Reference metric Christoffel symbols Γ̂^i_{jk}
        hat_chris = background.hat_christoffel  # (N, 3, 3, 3)

        N = len(rho0)

        # Allocate output arrays
        conn_D = np.empty(N)
        conn_S = np.empty((N, SPACEDIM))
        conn_tau = np.empty(N)

        # Call Numba kernel
        compute_connection_terms_kernel(N, hat_chris, fD_U, fTau_U, fS_UD,
                                        conn_D, conn_S, conn_tau)

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
        with self.timer('total_rhs'):
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
            with self.timer('extraction_geometry'):
                self._extract_geometry(r, bssn_vars, spacetime_mode, background, grid)

            # Compute Fluxes at interfaces (contains reconstruction + riemann)
            F_D_face, F_S_face, F_tau_face = self._compute_interface_fluxes(rho0, v_U, pressure, r, eos, reconstructor, riemann_solver, bssn_vars)

            # FLUX DERIVATIVE
            div_D, div_S, div_tau = self._compute_flux_derivative(F_D_face, F_S_face, F_tau_face)

            # CONNECTION TERMS FLUX derivative
            conn_D, conn_S, conn_tau = self._compute_connection_terms(rho0, v_U, pressure, W, h, background)

            # SOURCE TERMS
            with self.timer('source_terms'):
                src_S, src_tau = self._compute_source_terms(rho0, v_U, pressure, W, h, bssn_vars, bssn_d1,background, spacetime_mode, r)

            # ====================================================================
            # ASSEMBLE RHS: divergence + connection + source
            # ====================================================================
            rhs_D = np.zeros(N)
            rhs_S = np.zeros((N, SPACEDIM))
            rhs_tau = np.zeros(N)

            # Interior cells only (direct slicing is faster than boolean masking)
            g = NUM_GHOSTS
            rhs_D[g:N-g] = div_D[g:N-g] + conn_D[g:N-g]
            rhs_S[g:N-g, :] = div_S[g:N-g, :] + conn_S[g:N-g, :] + src_S[g:N-g, :]
            rhs_tau[g:N-g] = div_tau[g:N-g] + conn_tau[g:N-g] + src_tau[g:N-g]


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
        Compute partial fluxes at cell interfaces.

        GENERAL 3D STRUCTURE, 1D EVOLUTION:
        ====================================
        - Velocity v_U has shape (N, 3) with all spatial components
        - Only first spatial direction (index 0) fluxes computed via Riemann solver
        - Other spatial directions (indices 1,2) have zero flux

        Coordinate interpretation depends on background:
            - Spherical: v_U = [v^r, v^θ, v^φ], evolves v^r flux only

        This is a 1D solver limitation, NOT a physics assumption.
        All tensor algebra (T^μν, sources, connections) remains fully 3D.
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
        with self.timer('reconstruction'):
            (rhoL, vL, pL), (rhoR, vR, pR) = reconstructor.reconstruct_primitive_variables(
                rho0, v_r, pressure, x=r, boundary_type=recon_boundary
            )
        # ============================================================
        # Apply floors to reconstructed primitives
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
        with self.timer('cons2prim'):
            UL_D, UL_S_r, UL_tau = prim_to_cons(rhoL, vL, pL, geom_f, eos)
            UR_D, UR_S_r, UR_tau = prim_to_cons(rhoR, vR, pR, geom_f, eos)

        # Package for Riemann solver
        UL_batch = np.stack([UL_D, UL_S_r, UL_tau], axis=1)
        UR_batch = np.stack([UR_D, UR_S_r, UR_tau], axis=1)
        primL_batch = np.stack([rhoL, vL, pL], axis=1)
        primR_batch = np.stack([rhoR, vR, pR], axis=1)

        # Solve Riemann problem to get physical fluxes
        # Returns: [F_D, F_S_r, F_tau] in r spatial direction
        # Use fused kernel if available (faster), otherwise fallback to standard
        with self.timer('riemann_solver'):
            if hasattr(riemann_solver, 'solve_batch_fused'):
                F_phys_batch = riemann_solver.solve_batch_fused(UL_batch, UR_batch, primL_batch, primR_batch, geom_f, eos)
            else:
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

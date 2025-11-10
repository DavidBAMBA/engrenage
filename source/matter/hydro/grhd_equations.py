"""
GRHD Equations Module

This module provides the GRHDEquations class that orchestrates the complete
General Relativistic Hydrodynamics evolution equations using the Valencia formulation.

Classes:
    GRHDEquations: Main class for GRHD evolution
"""

import numpy as np
from typing import Tuple, Optional

from source.bssn.tensoralgebra import SPACEDIM, get_bar_gamma_LL, get_hat_D_bar_gamma_LL
from source.core.spacing import NUM_GHOSTS
from source.matter.hydro.geometry import ADMGeometry, ValenciaGeometry
from source.matter.hydro.stress_energy import StressEnergyTensor
from source.matter.hydro.cons2prim import prim_to_cons
from source.matter.hydro.atmosphere import AtmosphereParams


class GRHDEquations:
    """
    Orchestrates GRHD evolution equations in the Valencia formulation.

    This class implements the conservative evolution equations:
    ∂_t(U) + ∂_j(F̃^j) = S + connection_divergence_terms

    where:
    - U = (D, S_i, τ) are the conservative variables
    - F̃^j are the densitized fluxes
    - S are geometric source terms
    - connection are Christoffel symbol contributions
    """

    def __init__(self, eos, atmosphere: Optional[AtmosphereParams] = None,
                 boundary_mode: str = "parity"):
        """
        Initialize GRHD equations.

        Parameters
        ----------
        eos : IdealGasEOS, PolytropicEOS
            Equation of state
        atmosphere : AtmosphereParams, optional
            Atmosphere configuration
        boundary_mode : str
            "parity" for reflecting at r=0, "outflow" for zero-gradient
        """
        self.eos = eos
        self.atmosphere = atmosphere if atmosphere is not None else AtmosphereParams()
        self.boundary_mode = boundary_mode

    def compute_rhs(self, D: np.ndarray, S_tildeD: np.ndarray, tau: np.ndarray,
                    rho0: np.ndarray, v_U: np.ndarray, pressure: np.ndarray,
                    W: np.ndarray, h: np.ndarray,
                    geometry: ValenciaGeometry,
                    bssn_vars, bssn_d1, background,
                    reconstructor, riemann_solver,
                    spacetime_mode: str, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute complete RHS for GRHD evolution equations.

        This is the main orchestration method that:
        1. Reconstructs primitives to cell faces
        2. Solves Riemann problem for fluxes
        3. Computes flux divergence
        4. Adds connection term contributions
        5. Adds geometric source terms

        Parameters
        ----------
        D, S_tildeD, tau : np.ndarray
            Conservative variables
        rho0, v_U, pressure, W, h : np.ndarray
            Primitive variables and derived quantities
        geometry : ValenciaGeometry
            Geometric quantities
        bssn_vars, bssn_d1 : BSSN variables and derivatives
        background : Background metric
        reconstructor : Reconstruction method
        riemann_solver : Riemann solver
        spacetime_mode : str
            'fixed_minkowski', 'fixed', or 'dynamic'
        r : np.ndarray
            Radial coordinate

        Returns
        -------
        rhs_D, rhs_S, rhs_tau : np.ndarray
            Right-hand sides for conservative evolution
        """
        # CRITICAL: Enforce v^r = 0 at origin for spherical symmetry
        # Even tiny numerical drift (v^r ~ 1e-6) gets amplified by large
        # Christoffel symbols (Γ ~ 200/M) near origin, creating spurious
        # connection terms that act as mass sources and cause density to drift.
        if self.boundary_mode == "parity" and spacetime_mode != "fixed_minkowski":
            v_U[NUM_GHOSTS, 0] = 0.0  # Force v^r = 0 at r ≈ 0 (first interior cell)

            # CRITICAL: Recalculate W and h with the corrected v_U
            # The W passed from perfect_fluid was computed BEFORE this fix, so it's inconsistent.
            # Using inconsistent W in connection terms creates spurious mass sources.
            v_squared = np.einsum('xij,xi,xj->x', geometry.gamma_LL, v_U, v_U)
            W[NUM_GHOSTS] = 1.0 / np.sqrt(np.maximum(1.0 - v_squared[NUM_GHOSTS], 1e-16))

            # Recalculate h at origin (W changed, so h = 1 + eps + P/rho needs W for consistency)
            if self.eos is not None:
                eps_center = self.eos.eps_from_rho_p(rho0[NUM_GHOSTS], pressure[NUM_GHOSTS])
                h[NUM_GHOSTS] = 1.0 + eps_center + pressure[NUM_GHOSTS] / np.maximum(rho0[NUM_GHOSTS], 1e-30)

        # Compute stress-energy tensor components once for use in connection and source terms
        stress_energy = StressEnergyTensor(
            geometry=ADMGeometry(geometry.alpha, geometry.beta_U,
                                geometry.gamma_LL, geometry.gamma_UU),
            rho0=rho0, v_U=v_U, pressure=pressure, W=W, h=h
        )
        T00, T0i, Tij = stress_energy.compute_T4UU()
        _, T0_j, Ti_j = stress_energy.compute_T4UD()

        # Step 1: Reconstruct primitives and convert to conservatives
        UL, UR, primL, primR = self.reconstruct_and_convert(
            rho0, v_U[:, 0], pressure, r, reconstructor, geometry, spacetime_mode
        )

        # Step 2: Solve Riemann problem and get fluxes
        F_D, F_S, F_tau = self.solve_riemann_and_densitize(
            UL, UR, primL, primR, geometry, riemann_solver, spacetime_mode, bssn_vars
        )

        # Step 3: Compute divergence of fluxes

        # Step 3.1: Compute flux divergence (derivative term)
        rhs_D, rhs_S, rhs_tau = self.compute_divergence(F_D, F_S, F_tau, geometry.dr)

        # Step 3.2: Add connection term contributions from divergence
        # CRITICAL: Always compute connection terms, even in Minkowski spacetime.
        # In spherical coordinates (or any curvilinear system), the reference metric
        # Christoffel symbols Γ̂^i_{jk} are non-zero and provide essential geometric terms.
        # For example, the "hoop stress" term 2p/r in spherical blast waves comes from
        # connection terms, NOT from explicit source terms.
        conn_D, conn_S, conn_tau = self.compute_connection_terms(
            rho0, v_U, pressure, W, h, geometry, background.hat_christoffel,
            T0i=T0i, Ti_j=Ti_j
        )
        rhs_D += conn_D
        rhs_S += conn_S
        rhs_tau += conn_tau

        # Step 4: Add geometric source terms
        if spacetime_mode == 'dynamic':
            src_S, src_tau = self.compute_source_terms(
                rho0, v_U, pressure, W, h, geometry,
                bssn_vars, bssn_d1, background, spacetime_mode, r,
                T00=T00, T0i=T0i, Tij=Tij, T0_j=T0_j
            )
            rhs_S += src_S
            rhs_tau += src_tau

        return rhs_D, rhs_S, rhs_tau

    def compute_divergence(self, F_D_face: np.ndarray, F_S_face: np.ndarray,
                          F_tau_face: np.ndarray, dr: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute flux divergence, derivative term.

        Formula: -∂_r(F̃^r) = -(F̃^r_{i+1/2} - F̃^r_{i-1/2}) / Δr

        Parameters
        ----------
        F_D_face : np.ndarray
            Density flux at cell faces (N_faces,)
        F_S_face : np.ndarray
            Momentum flux at cell faces (N_faces, 3)
        F_tau_face : np.ndarray
            Energy flux at cell faces (N_faces,)
        dr : float
            Radial grid spacing

        Returns
        -------
        rhs_D, rhs_S, rhs_tau : np.ndarray
            RHS contributions from flux divergence
        """
        # Determine number of cells from number of faces
        N = F_D_face.shape[0] + 1

        # Initialize RHS arrays
        rhs_D = np.zeros(N)
        rhs_S = np.zeros((N, SPACEDIM))
        rhs_tau = np.zeros(N)

        # Compute divergence using vectorized operations for interior cells
        inv_dr = 1.0 / (dr + 1e-30)

        # Interior cells: NUM_GHOSTS <= i < N - NUM_GHOSTS
        i_start = NUM_GHOSTS
        i_end = N - NUM_GHOSTS

        rhs_D[i_start:i_end]    = -(F_D_face[i_start:i_end]    -  F_D_face[i_start-1:i_end-1])    * inv_dr
        rhs_S[i_start:i_end, :] = -(F_S_face[i_start:i_end, :] -  F_S_face[i_start-1:i_end-1, :]) * inv_dr
        rhs_tau[i_start:i_end]  = -(F_tau_face[i_start:i_end]  -  F_tau_face[i_start-1:i_end-1])  * inv_dr

        return rhs_D, rhs_S, rhs_tau

    def reconstruct_and_convert(self, rho0: np.ndarray, v_primary: np.ndarray,
                                pressure: np.ndarray, r: np.ndarray,
                                reconstructor, geometry: ValenciaGeometry,
                                spacetime_mode: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Reconstruct primitive variables and convert to conservatives at cell faces.

        Parameters
        ----------
        rho0 : np.ndarray
            Rest mass density at cell centers
        v_primary : np.ndarray
            Primary velocity component (v^r in spherical)
        pressure : np.ndarray
            Pressure at cell centers
        r : np.ndarray
            Radial coordinate array
        reconstructor : Reconstruction
            Reconstruction method object
        geometry : ValenciaGeometry
            Geometry with metric at faces
        spacetime_mode : str
            'fixed_minkowski', 'fixed', or 'dynamic'

        Returns
        -------
        UL_batch, UR_batch : np.ndarray
            Left/right conservative states at faces (N_faces, 3)
        primL_batch, primR_batch : np.ndarray
            Left/right primitive states at faces (N_faces, 3)
        """
        # Determine boundary condition type for reconstruction
        # Determine boundary type for reconstruction
        # CRITICAL: Use reflecting ONLY for parity boundaries in non-Minkowski spacetime
        # In Minkowski (spacetime_mode=="fixed_minkowski"), always use outflow to avoid
        # spurious reflections that contaminate the Sod shock test
        use_reflecting = (self.boundary_mode == "parity" and spacetime_mode != "fixed_minkowski")
        recon_boundary = "reflecting" if use_reflecting else "outflow"

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
                rho_ref = rho0[NUM_GHOSTS]
                p_ref = pressure[NUM_GHOSTS]
                rhoL[0] = rho_ref
                rhoR[0] = rho_ref
                pL[0] = p_ref
                pR[0] = p_ref

        # Interpolate metric to cell faces (arithmetic average)
        # geometry.gamma_LL has shape (N, 3, 3), we need (N-1, 3, 3) at faces
        gamma_LL_f = 0.5 * (geometry.gamma_LL[:-1] + geometry.gamma_LL[1:])  # (N-1, 3, 3)
        gamma_primary_primary_f = gamma_LL_f[:, 0, 0]  # γ_rr at faces (N-1,)

        # Apply physical limiters if available
        if hasattr(reconstructor, "apply_physical_limiters"):
            (rhoL, vL, pL), (rhoR, vR, pR) = reconstructor.apply_physical_limiters(
                (rhoL, vL, pL), (rhoR, vR, pR),
                atmosphere=self.atmosphere,
                gamma_rr=gamma_primary_primary_f  # Now has correct shape (N-1,)
            )

        # Convert primitives to conservatives at interfaces
        UL_D, UL_S_primary, UL_tau = prim_to_cons(
            rhoL, vL, pL, gamma_primary_primary_f, self.eos
        )
        UR_D, UR_S_primary, UR_tau = prim_to_cons(
            rhoR, vR, pR, gamma_primary_primary_f, self.eos
        )

        # Package for Riemann solver
        UL_batch = np.stack([UL_D, UL_S_primary, UL_tau], axis=1)
        UR_batch = np.stack([UR_D, UR_S_primary, UR_tau], axis=1)
        primL_batch = np.stack([rhoL, vL, pL], axis=1)
        primR_batch = np.stack([rhoR, vR, pR], axis=1)

        return UL_batch, UR_batch, primL_batch, primR_batch

    def solve_riemann_and_densitize(self, UL_batch: np.ndarray, UR_batch: np.ndarray,
                                    primL_batch: np.ndarray, primR_batch: np.ndarray,
                                    geometry: ValenciaGeometry, riemann_solver,
                                    spacetime_mode: str, bssn_vars) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve Riemann problem and apply densitization.

        Parameters
        ----------
        UL_batch, UR_batch : np.ndarray
            Conservative states at faces
        primL_batch, primR_batch : np.ndarray
            Primitive states at faces
        geometry : ValenciaGeometry
            Geometry with lapse, shift, metric at faces
        riemann_solver : RiemannSolver
            Riemann solver object
        spacetime_mode : str
            'fixed_minkowski', 'fixed', or 'dynamic'

        Returns
        -------
        F_D_face, F_S_tildeD_face, F_tau_face : np.ndarray
            Densitized fluxes at cell faces
        """
        # Interpolate geometry to cell faces (arithmetic average)
        # All geometry arrays have shape (N,), we need (N-1,) at faces
        alpha_f = 0.5 * (geometry.alpha[:-1] + geometry.alpha[1:])
        beta_U_f = 0.5 * (geometry.beta_U[:-1] + geometry.beta_U[1:])
        beta_primary_f = beta_U_f[:, 0]  # Extract primary component
        gamma_LL_f = 0.5 * (geometry.gamma_LL[:-1] + geometry.gamma_LL[1:])
        gamma_primary_primary_f = gamma_LL_f[:, 0, 0]

        # CRITICAL FIX: Interpolate φ to faces FIRST, then compute exp(6φ)
        # This ensures exact cancellation with connection terms
        # Original: exp(6·φ_face), NOT 0.5*(e6φ_i + e6φ_{i+1})
        phi_arr = np.asarray(bssn_vars.phi, dtype=float)
        phi_face = 0.5 * (phi_arr[:-1] + phi_arr[1:])
        e6phi_f = np.exp(6.0 * phi_face)

        # Solve Riemann problem to get physical fluxes
        F_phys_batch = riemann_solver.solve_batch(
            UL_batch, UR_batch, primL_batch, primR_batch,
            gamma_primary_primary_f, alpha_f, beta_primary_f, self.eos
        )

        # Densitization: F̃ = α e^{6φ} F^phys
        dens_factor = alpha_f * e6phi_f
        F_batch = dens_factor[:, None] * F_phys_batch

        # Enforce zero momentum flux at the r≈0 interface (spherical boundary)
        # This is the last guard against numerical drift creating spurious fluxes at origin
        if self.boundary_mode == "parity" and spacetime_mode != "fixed_minkowski":
            if len(F_batch) > 0:
                F_batch[0, 1] = 0.0  # F_Sr = 0 at the r≈0 interface

        # Additional freeze: zero fluxes at atmosphere interfaces
        if spacetime_mode != "fixed_minkowski":
            try:
                rho_atm_face = self.atmosphere.rho_floor * getattr(self.atmosphere, 'rho_threshold', 1.0)
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

    def compute_connection_terms(self, rho0: np.ndarray, v_U: np.ndarray,
                                pressure: np.ndarray, W: np.ndarray, h: np.ndarray,
                                geometry: ValenciaGeometry,
                                hat_chris: np.ndarray,
                                T0i: np.ndarray, Ti_j: np.ndarray,
                                return_debug: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Christoffel connection contributions to RHS.

        Connection form ( +):
            D, τ:   -Γ̂^k_{ki} F̃^i
            S_i:    -Γ̂^k_{kj} F̃^j_i + Γ̂^l_{ji} F̃^j_l

        Parameters
        ----------
        rho0, v_U, pressure, W, h : np.ndarray
            Hydrodynamic variables
        geometry : ValenciaGeometry
            Geometric quantities
        hat_chris : np.ndarray
            Reference Christoffel symbols (N, 3, 3, 3)
        T0i : np.ndarray
            Contravariant stress-energy T^{0i} components (N, 3)
        Ti_j : np.ndarray
            Mixed stress-energy T^i_j components (N, 3, 3)

        Returns
        -------
        conn_D, conn_S, conn_tau : np.ndarray
            Connection contributions to RHS
        """
        N = len(rho0)
        alpha = geometry.alpha
        e6phi = geometry.e6phi

        # Conservative density flux rho_star = e^{6φ} ρ₀ W
        # Note: u^0 = W/α, so α*u^0 = W
        rho_star = e6phi * rho0 * W

        # Physical fluxes at cell centers (partial flux vectors)
        fD_U = rho_star[:, np.newaxis] * v_U

        # Energy (tau) partial flux vector using T^{0j} from stress_energy tensor
        # fTau_U = α² e^{6φ} T^{0j} - rho_star v^j
        fTau_U = (alpha ** 2 * e6phi)[:, np.newaxis] * T0i - rho_star[:, np.newaxis] * v_U

        # Momentum partial flux tensor F̃^j_i = α e^{6φ} T^j_i
        # Use precomputed mixed tensor T^i_j from stress_energy module
        F_S = (alpha * e6phi)[:, np.newaxis, np.newaxis] * Ti_j

        Gamma_trace = np.einsum('xkkj->xj', hat_chris)

        # Connection contributions
        conn_D   = - np.einsum('xj,xj->x', Gamma_trace, fD_U)
        conn_tau = - np.einsum('xj,xj->x', Gamma_trace, fTau_U)
        conn_S   = (
                   - np.einsum('xl,xli->xi', Gamma_trace, F_S)
                   + np.einsum('xlji,xjl->xi', hat_chris, F_S)
        )

        if not return_debug:
            return conn_D, conn_S, conn_tau

        debug = {
            "rho_star_flux": fD_U.copy(),
            "tau_flux": fTau_U.copy(),
            "momentum_flux": F_S.copy(),
            "Gamma_trace": Gamma_trace.copy(),
        }
        return conn_D, conn_S, conn_tau, debug

    def compute_source_terms(self, rho0: np.ndarray, v_U: np.ndarray,
                            pressure: np.ndarray, W: np.ndarray, h: np.ndarray,
                            geometry: ValenciaGeometry,
                            bssn_vars, bssn_d1, background,
                            spacetime_mode: str, r: np.ndarray,
                            T00: np.ndarray, T0i: np.ndarray, Tij: np.ndarray, T0_j: np.ndarray,
                            return_debug: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute geometric source terms for Valencia equations.

        Source terms arise from:
        1. Extrinsic curvature K_ij contractions
        2. Lapse gradient ∂_i α
        3. Shift covariant derivative ∇̂_i β^j
        4. Metric covariant derivative ∇̂_i γ_{jk}

        Parameters
        ----------
        rho0, v_U, pressure, W, h : np.ndarray
            Hydrodynamic variables
        geometry : ValenciaGeometry
            Geometric quantities
        bssn_vars, bssn_d1 : BSSN variables and derivatives
        background : Background metric
        spacetime_mode : str
            'fixed' or 'dynamic'
        r : np.ndarray
            Radial coordinate
        T00 : np.ndarray
            Contravariant stress-energy T^{00} component (N,)
        T0i : np.ndarray
            Contravariant stress-energy T^{0i} components (N, 3)
        Tij : np.ndarray
            Contravariant stress-energy T^{ij} components (N, 3, 3)
        T0_j : np.ndarray
            Mixed stress-energy T^0_j components (N, 3)

        Returns
        -------
        src_S, src_tau : np.ndarray
            Source terms for momentum and energy
        """
        N = len(rho0)

        # Fixed spacetime has no source terms
        if spacetime_mode == 'fixed':
            src_S_vector = np.zeros((N, SPACEDIM))
            src_tau = np.zeros(N)
            if not return_debug:
                return src_S_vector, src_tau

            zero_term_vec = np.zeros((N, SPACEDIM))
            zero_term_scalar = np.zeros(N)
            zero_hat = np.zeros((N, SPACEDIM, SPACEDIM))
            debug_fixed = {
                "energy_Kij": zero_term_scalar.copy(),
                "energy_dalpha": zero_term_scalar.copy(),
                "energy_total": zero_term_scalar.copy(),
                "momentum_T00_alpha": zero_term_vec.copy(),
                "momentum_T0j_beta": zero_term_vec.copy(),
                "momentum_metric": zero_term_vec.copy(),
                "momentum_total": zero_term_vec.copy(),
                "hatD_beta": zero_hat.copy(),
            }
            return src_S_vector, src_tau, debug_fixed

        # Extract needed quantities
        alpha = geometry.alpha
        beta_U = geometry.beta_U
        gamma_LL = geometry.gamma_LL
        gamma_UU = geometry.gamma_UU
        e6phi = geometry.e6phi

        phi = np.asarray(bssn_vars.phi, dtype=float)
        e4phi = np.exp(4.0 * phi)

        # For source terms, we use precomputed mixed components T^0_j from stress_energy module
        T0i_UD = T0_j

        # ====================================================================
        # ENERGY SOURCE TERM
        # ====================================================================
        # S_τ = α e^{6φ} [K_ij stress_block - T^0_μ ∂_μ α]

        # Get BSSN quantities
        h_LL = bssn_vars.h_LL

        K = np.asarray(bssn_vars.K, dtype=float)
        a_LL = bssn_vars.a_LL

        # Physical extrinsic curvature: K_ij = e^{4φ} Ā_ij + (1/3) γ_ij K
        K_LL = e4phi[:, np.newaxis, np.newaxis] * a_LL + (1.0/3.0) * gamma_LL * K[:, np.newaxis, np.newaxis]

        # Stress block: T^00 β^i β^j + 2 T^{0i} β^j + T^{ij} 
        stress_block = (T00[:, None, None] * np.einsum('xi,xj->xij', beta_U, beta_U) +
                       2.0 * np.einsum('xi,xj->xij', T0i, beta_U) +
                       Tij)

        # Term 1: K_ij × stress_block
        term1_tau = np.einsum('nij,nij->n', K_LL, stress_block)

        # Term 2: Lapse gradient 
        d_alpha = np.asarray(bssn_d1.lapse, dtype=float)
        term2_tau = -(np.einsum('x,xi,xi->x', T00, beta_U, d_alpha) +
                     np.einsum('xi,xi->x', T0i, d_alpha))

        src_tau = alpha * e6phi * (term1_tau + term2_tau)

        # ====================================================================
        # MOMENTUM SOURCE TERM
        # ====================================================================
        # S_i = α e^{6φ} [-T^{00} ∂_i α + T^0_j ∇̂_i β^j + 0.5 stress_block × ∇̂_i γ_{jk}]

        # Prepare momentum term containers
        first_term = -T00[:, None] * alpha[:, None] * d_alpha
        second_term = np.zeros((N, SPACEDIM))
        third_term = np.zeros((N, SPACEDIM))

        # Term 2: T^0_j ∇̂_i β^j (shift covariant derivative)
        # ∇̂_i β^j = ∂_i β^j + Γ̂^j_{ik} β^k
        # CRITICAL: Apply product rule for β^j = shift_rescaled^j × inverse_scaling^j
        # ∂_i β^j = inverse_scaling^j × ∂_i(shift_rescaled^j) + shift_rescaled^j × ∂_i(inverse_scaling^j)

        # Get shift derivatives from BSSN
        shift_d1 = np.asarray(bssn_d1.shift_U, dtype=float)  # ∂_j(shift_rescaled^i)

        # Get rescaled shift from BSSN (to apply product rule)
        shift_rescaled = np.asarray(bssn_vars.shift_U, dtype=float)
        # Handle 1D case (only radial component)
        if shift_rescaled.ndim == 1:
            shift_rescaled_2d = np.zeros((N, SPACEDIM))
            shift_rescaled_2d[:, 0] = shift_rescaled
            shift_rescaled = shift_rescaled_2d

        # Apply product rule: ∂_i β^j = s^j ∂_i shift^j + shift^j ∂_i s^j
        dbeta_dx = (background.inverse_scaling_vector[:, :, None] * shift_d1 +
                   shift_rescaled[:, :, None] * background.d1_inverse_scaling_vector)

        hat_chris = background.hat_christoffel
        hatD_beta = np.zeros((N, SPACEDIM, SPACEDIM))

        # Vectorized computation of shift covariant derivative: ∇̂_i β^j = ∂_i β^j + Γ̂^j_{ik} β^k
        hatD_beta = dbeta_dx + np.einsum('xjik,xk->xij', hat_chris, beta_U)
        # Vectorized computation of second term: sum_j T^0_j * ∇̂_i β^j
        second_term = np.einsum('xj,xij->xi', T0i_UD, hatD_beta)

        # Term 3: 0.5 × stress_block × ∇̂_i γ_{jk}
        # ∇̂_i γ_{jk} = e^{4φ} (∂_i γ̄_{jk} + 4 γ̄_{jk} ∂_i φ)
        # where ∂_i γ̄_{jk} = Γ̂_{jik} + Γ̂_{kij} from reference metric
        d_phi = np.asarray(bssn_d1.phi, dtype=float)

        # CRITICAL FIX: Compute bar_gamma_LL (conformal metric)
        bar_gamma_LL = get_bar_gamma_LL(r, h_LL, background)


        d1_h = bssn_d1.h_LL
        hat_D_gamma = get_hat_D_bar_gamma_LL(r, h_LL, d1_h, background)

        # hat_D_gamma has shape (N, 3, 3, 3) with indices [x, m, n, p] = ∇̂_p γ̄_{mn}
        # To get ∇̂_i γ̄_{jk}, we need hat_D_gamma[:, j, k, i], transpose to (N, i, j, k)
        # CRITICAL FIX: Use bar_gamma_LL (full conformal metric), not h_LL (deviation)
        # Formula: ∇̂_i γ_{jk} = e^{4φ} (∂_i γ̄_{jk} + 4 γ̄_{jk} ∂_i φ)
        hat_D_gamma_reordered = np.moveaxis(hat_D_gamma, 3, 1)  # (N, j, k, i) -> (N, i, j, k)
        cov_deriv_gamma = (e4phi[:, None, None, None] *
                          (hat_D_gamma_reordered +
                           4.0 * bar_gamma_LL[:, None, :, :] * d_phi[:, :, None, None]))
        third_term = 0.5 * np.einsum('xjk,xijk->xi', stress_block, cov_deriv_gamma)

        total_momentum_term = first_term + second_term + third_term
        src_S_vector = alpha[:, None] * e6phi[:, None] * total_momentum_term

        if not return_debug:
            return src_S_vector, src_tau

        debug = {
            "energy_Kij": term1_tau.copy(),
            "energy_dalpha": term2_tau.copy(),
            "energy_total": src_tau.copy(),
            "momentum_T00_alpha": first_term.copy(),
            "momentum_T0j_beta": second_term.copy(),
            "momentum_metric": third_term.copy(),
            "momentum_total": src_S_vector.copy(),
            "hatD_beta": hatD_beta.copy(),
        }
        return src_S_vector, src_tau, debug


# ============================================================================
# CONSERVATIVE VARIABLES: DENSITIZED FORMULATION
# ============================================================================

def compute_densitized_conservatives_from_primitives(
    rho0: np.ndarray,
    v_U: np.ndarray,
    W: np.ndarray,
    h: np.ndarray,
    pressure: np.ndarray,
    alpha: np.ndarray,
    e6phi: np.ndarray,
    gamma_LL: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute densitized conservative variables from primitives (3D formulation).

    This implements the Valencia formulation where conservative variables are
    defined as densitized quantities from the start:

    D̃ = e^{6φ} W ρ₀
    S̃ᵢ = e^{6φ} W ρ₀ h vᵢ
    τ̃ = e^{6φ} (α² T⁰⁰ - W ρ₀)

    where T⁰⁰ = ρ₀ h W² - P is the time-time component of the stress-energy tensor.

    Parameters
    ----------
    rho0 : np.ndarray
        Rest mass density (N,)
    v_U : np.ndarray
        Contravariant velocity (N, 3)
    W : np.ndarray
        Lorentz factor (N,)
    h : np.ndarray
        Specific enthalpy (N,)
    pressure : np.ndarray
        Pressure (N,)
    alpha : np.ndarray
        Lapse function (N,)
    e6phi : np.ndarray
        Densitization factor e^{6φ} (N,)
    gamma_LL : np.ndarray
        Spatial metric (N, 3, 3)

    Returns
    -------
    D_tilde : np.ndarray
        Densitized rest mass density (N,)
    S_tildeD : np.ndarray
        Densitized momentum density (N, 3)
    tau_tilde : np.ndarray
        Densitized energy density (N,)
    """
    # D̃ = e^{6φ} W ρ₀
    D_tilde = e6phi * W * rho0

    # S̃ᵢ = e^{6φ} W² ρ₀ h vᵢ
    # Lower the index: vᵢ = γᵢⱼ vʲ (using Einstein summation)
    # Handle both scalar and array cases
    if gamma_LL.ndim == 2:
        # Scalar case: gamma_LL shape is (3, 3)
        v_D = np.einsum('ij,j->i', gamma_LL, v_U)  # vᵢ = γᵢⱼ vʲ
        S_tildeD = (e6phi * W * W * rho0 * h) * v_D
    else:
        # Array case: gamma_LL shape is (N, 3, 3)
        v_D = np.einsum('xij,xj->xi', gamma_LL, v_U)  # vᵢ = γᵢⱼ vʲ
        S_tildeD = (e6phi * W * W * rho0 * h)[:, np.newaxis] * v_D

    # τ̃ = e^{6φ} (α² T⁰⁰ - W ρ₀)
    # where T⁰⁰ = ρ₀ h W² - P
    T00 = rho0 * h * W * W - pressure
    tau_tilde = e6phi * (alpha**2 * T00 - W * rho0)

    return D_tilde, S_tildeD, tau_tilde


def dedensitize_conservatives(
    D_tilde: np.ndarray,
    S_tildeD: np.ndarray,
    tau_tilde: np.ndarray,
    e6phi: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    De-densitize conservative variables: U = Ũ / e^{6φ}

    This converts from the densitized conservative variables used in evolution
    to the physical (non-densitized) conservative variables needed for cons2prim.

    Parameters
    ----------
    D_tilde : np.ndarray
        Densitized rest mass density (N,)
    S_tildeD : np.ndarray
        Densitized momentum density (N, 3) or (N,) for 1D
    tau_tilde : np.ndarray
        Densitized energy density (N,)
    e6phi : np.ndarray
        Densitization factor e^{6φ} (N,)

    Returns
    -------
    D : np.ndarray
        Physical rest mass density (N,)
    S_D : np.ndarray
        Physical momentum density (N, 3) or (N,)
    tau : np.ndarray
        Physical energy density (N,)
    """
    inv_e6phi = 1.0 / np.maximum(e6phi, 1e-30)

    D = D_tilde * inv_e6phi

    # Handle both 1D and 3D momentum
    if S_tildeD.ndim == 1:
        S_D = S_tildeD * inv_e6phi
    else:
        S_D = S_tildeD * inv_e6phi[:, np.newaxis]

    tau = tau_tilde * inv_e6phi

    return D, S_D, tau
